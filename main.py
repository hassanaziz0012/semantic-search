import csv
import re
import os
import pickle
import logging
from dataclasses import dataclass
from typing import List
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "thenlper/gte-large"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_model() -> SentenceTransformer:
    # Optimization: Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model: {MODEL_NAME} on device: {device}")

    try:
        return SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        logging.error(f"Failed to load model from {MODEL_NAME}.")
        raise e

@dataclass
class Movie:
    release_year: int
    title: str
    origin_ethnicity: str
    director: str
    cast: str
    genre: str
    plot: str

def divide_into_chunks(text: str, chunk_size: int = 3, overlap: int = 1) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(sentences), step):
        if i > 0 and i + chunk_size > len(sentences) + step:
             pass

        chunk = sentences[i : i + chunk_size]
        chunks.append(" ".join(chunk))

        if i + chunk_size >= len(sentences):
            break

    return chunks

def load_dataset(file_path: str = "data/wiki_movie_plots_deduped.csv") -> List[Movie]:
    movies = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            movies.append(Movie(
                release_year=int(row['Release Year']),
                title=row['Title'],
                origin_ethnicity=row['Origin/Ethnicity'],
                director=row['Director'],
                cast=row['Cast'],
                genre=row['Genre'],
                plot=row['Plot']
            ))
    return movies

def convert_to_text(movie: Movie) -> List[str]:
    chunks = divide_into_chunks(movie.plot)
    if not chunks:
        # If no plot or empty, just return one "chunk" with empty plot?
        # Or return empty list?
        # Let's return a list with one entry capable of empty plot.
        chunks = [""]

    text_chunks = []
    for i, chunk in enumerate(chunks):
        text = (
            f"Title: {movie.title}\n"
            f"Year: {movie.release_year}\n"
            f"Director: {movie.director}\n"
            f"Cast: {movie.cast}\n"
            f"Genre: {movie.genre}\n"
            f"Origin: {movie.origin_ethnicity}\n\n"
            f"Plot ({i+1}):\n"
            f"{chunk}"
        )
        text_chunks.append(text)
    return text_chunks

def load_or_create_embeddings(movies: List[Movie], model: SentenceTransformer) -> dict:
    cache_path = "drive/MyDrive/semantic-search/cache/embeddings.pkl"

    if os.path.exists(cache_path):
        logging.info("Loading embeddings from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logging.info("Cache not found. Generating embeddings...")

    # 1. Convert movies to text chunks
    logging.info("Converting movies to text chunks...")
    all_texts = []
    chunk_metadata = [] # Keep track of which movie/chunk each text belongs to

    for movie in movies:
        chunks = convert_to_text(movie)
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            chunk_metadata.append({
                "movie_title": movie.title,
                "chunk_index": i,
                "release_year": movie.release_year
            })

    # 2. Encode
    logging.info(f"Encoding {len(all_texts)} text chunks...")
    # Optimization: Batch processing
    embeddings = model.encode(all_texts, batch_size=156, show_progress_bar=True, convert_to_numpy=True)

    # 3. Save to cache
    logging.info("Saving embeddings to cache...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    data = {
        "embeddings": embeddings,
        "texts": all_texts,
        "metadata": chunk_metadata
    }

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    logging.info("Embeddings saved.")
    return data

def search(query: str, model: SentenceTransformer, embedding_data: dict, top_k: int = 20) -> List[dict]:
    """
    Search for movies matching the query string.
    """
    logging.info(f"Searching for: '{query}'")

    # 1. Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 2. Get embeddings from data
    corpus_embeddings = embedding_data["embeddings"]

    # Ensure corpus_embeddings is a tensor
    if not isinstance(corpus_embeddings, torch.Tensor):
        corpus_embeddings = torch.tensor(corpus_embeddings)

    # Move to same device as query_embedding if needed
    if query_embedding.device != corpus_embeddings.device:
        corpus_embeddings = corpus_embeddings.to(query_embedding.device)

    scores = torch.cosine_similarity(query_embedding, corpus_embeddings, dim=1)

    # faster and more efficient than sorted()[:top_k]
    top_results = torch.topk(scores, k=min(top_k, len(scores)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        results.append({
            "score": score.item(),
            "text": embedding_data["texts"][idx],
            "metadata": embedding_data["metadata"][idx]
        })

    return results

def display_search_results(query: str, results: List[dict]):
    logging.info(f"\n--- Top 20 Results for '{query}' ---")
    seen_movies = set()
    rank = 0
    
    for result in results:
        movie_identifier = (result['metadata']['movie_title'], result['metadata']['release_year'])
        
        if movie_identifier in seen_movies:
            continue
        
        seen_movies.add(movie_identifier)
        rank += 1
        
        # Parse details from text
        text = result['text']
        
        genre = "Unknown"
        genre_match = re.search(r"Genre: (.*)", text)
        if genre_match:
            genre = genre_match.group(1).strip()
        
        director = "Unknown"
        director_match = re.search(r"Director: (.*)", text)
        if director_match:
            director = director_match.group(1).strip()
        
        description = text
        plot_match = re.search(r"Plot \(\d+\):\n(.*)", text, re.DOTALL)
        if plot_match:
            description = plot_match.group(1).strip()
        
        if len(description) > 200:
            description = description[:200] + "..."
        
        description = description.replace('\n', ' ').strip()
        
        print(f"{rank}) {result['metadata']['movie_title']} ({result['score']:.4f}) [{genre}, {result['metadata']['release_year']}, {director}]")
        print(f"{description}")
        print("\n---- ---- ----\n")

        if rank >= 20:
            break

def main():
    try:
        movies = load_dataset()
        logging.info(f"Successfully loaded {len(movies)} movies.")

        if movies:
            model = load_model()
            embedding_data = load_or_create_embeddings(movies, model)
            logging.info(f"Loaded/Generated {len(embedding_data['embeddings'])} embeddings.")

            query = input("Enter your search query: ")
            # Search with a larger top_k to likely get enough unique movies
            results = search(query, model, embedding_data, top_k=200)

            display_search_results(query, results)


    except FileNotFoundError:
        logging.info("Error: Dataset file not found.")
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
