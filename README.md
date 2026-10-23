# Semantic search

A movie search that finds films by what you mean instead of the words you type. Search for "a wizard school" and you get Harry Potter, even though no title says that.

It embeds the plots of 34,886 movies with the [thenlper/gte-large](https://huggingface.co/thenlper/gte-large) embedding model and ranks every search by cosine similarity against those vectors. The whole thing is one file, `main.py`.

I explain how it works, step by step, in [How semantic search works, with a Python example](https://www.hassandev.me/blog/youtube/how-semantic-search-works-MKUXfWmhBXc). There's a video of the theory on the same page.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.13 or newer.

```bash
git clone https://github.com/hassanaziz0012/semantic-search.git
cd semantic-search
uv sync
```

Download the [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset from Kaggle and put `wiki_movie_plots_deduped.csv` in a `data` folder.

## Run it

```bash
uv run main.py
```

The first run downloads gte-large (about 670 MB) and embeds every plot, saving the vectors to `cache/embeddings.pkl`. Later runs load that file and go straight to the search prompt. Type a query and press Enter, or press Enter on a blank line to quit.

```text
Search (blank line to quit): fantasy movies
1) The Lord of the Rings: The Two Towers (0.8742) [adventure, fantasy, 2002, Peter Jackson]
2) The Chronicles of Narnia: Prince Caspian (0.8737) [fantasy, 2008, Andrew Adamson]
3) The Lord of the Rings: The Return of the King (0.8718) [adventure, fantasy, 2003, Peter Jackson]
```

That output is from a 350-movie slice of the dataset, with the plot line under each result cut.

## Without a GPU

The first run is slow on a CPU. The full dataset is 328,648 chunks, and on my laptop's CPU gte-large embedded fewer than two a second, so the whole set would take more than two days. The code uses an NVIDIA GPU on its own if there is one, and Google Colab has free GPU runtimes.

On a CPU, switch `MODEL_NAME` to the commented-out `sentence-transformers/all-MiniLM-L6-v2` line. It ran about 20 times faster in my test, with somewhat worse results. Delete `cache/embeddings.pkl` whenever you change the model or the data.
