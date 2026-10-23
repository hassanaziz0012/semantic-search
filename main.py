import csv
import os
import pickle
import re
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "thenlper/gte-large"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_PATH = "data/wiki_movie_plots_deduped.csv"
CACHE_PATH = "cache/embeddings.pkl"


@dataclass
class Movie:
    release_year: int
    title: str
    origin: str
    director: str
    cast: str
    genre: str
    plot: str


def load_model() -> SentenceTransformer:
    # Half precision is fast on a GPU but very slow on a CPU
    if torch.cuda.is_available():
        return SentenceTransformer(MODEL_NAME, device="cuda", model_kwargs={"dtype": torch.float16})
    return SentenceTransformer(MODEL_NAME, device="cpu", model_kwargs={"dtype": torch.float32})


def load_dataset(file_path: str = DATASET_PATH) -> list[Movie]:
    movies = []
    with open(file_path, encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            movies.append(Movie(
                release_year=int(row["Release Year"]),
                title=row["Title"],
                origin=row["Origin/Ethnicity"],
                director=row["Director"],
                cast=row["Cast"],
                genre=row["Genre"],
                plot=row["Plot"],
            ))
    return movies


def divide_into_chunks(text: str, chunk_size: int = 3, overlap: int = 1) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunks.append(" ".join(sentences[i : i + chunk_size]))
        if i + chunk_size >= len(sentences):
            break

    return chunks


def convert_to_text(movie: Movie) -> list[str]:
    # Every chunk repeats the movie's details, so each one makes sense on its own
    chunks = divide_into_chunks(movie.plot) or [""]
    return [
        f"Title: {movie.title}\n"
        f"Year: {movie.release_year}\n"
        f"Director: {movie.director}\n"
        f"Cast: {movie.cast}\n"
        f"Genre: {movie.genre}\n"
        f"Origin: {movie.origin}\n\n"
        f"Plot ({i + 1}):\n"
        f"{chunk}"
        for i, chunk in enumerate(chunks)
    ]


def load_or_create_embeddings(movies: list[Movie], model: SentenceTransformer) -> dict:
    if os.path.exists(CACHE_PATH):
        print("Loading embeddings from cache...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    texts = []
    metadata = []
    for movie in movies:
        for i, text in enumerate(convert_to_text(movie)):
            texts.append(text)
            metadata.append({
                "title": movie.title,
                "year": movie.release_year,
                "genre": movie.genre,
                "director": movie.director,
                "chunk_index": i,
            })

    print(f"Embedding {len(texts)} chunks from {len(movies)} movies...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    data = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f)

    return data


def search(query: str, model: SentenceTransformer, embedding_data: dict, top_k: int = 200) -> list[dict]:
    # The query goes through the same model as the dataset
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = torch.tensor(embedding_data["embeddings"], device=query_embedding.device)

    scores = torch.cosine_similarity(query_embedding, corpus_embeddings, dim=1)
    top_results = torch.topk(scores, k=min(top_k, len(scores)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        results.append({
            "score": score.item(),
            "text": embedding_data["texts"][idx],
            "metadata": embedding_data["metadata"][idx],
        })
    return results


def display_search_results(results: list[dict], limit: int = 20):
    # One movie can match with several chunks, so show each movie once
    seen = set()
    for result in results:
        meta = result["metadata"]
        if (meta["title"], meta["year"]) in seen:
            continue
        seen.add((meta["title"], meta["year"]))

        plot = result["text"].split(":\n", maxsplit=1)[-1].replace("\n", " ")
        if len(plot) > 200:
            plot = plot[:200] + "..."

        print(f"{len(seen)}) {meta['title']} ({result['score']:.4f}) [{meta['genre']}, {meta['year']}, {meta['director']}]")
        print(plot)
        print()

        if len(seen) >= limit:
            break


def main():
    movies = load_dataset()
    print(f"Loaded {len(movies)} movies.")

    model = load_model()
    embedding_data = load_or_create_embeddings(movies, model)

    while query := input("Search (blank line to quit): ").strip():
        display_search_results(search(query, model, embedding_data))


if __name__ == "__main__":
    main()
