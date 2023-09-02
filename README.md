# Script for Translating and Updating DialogSum Dataset

This script is designed to translate and update a dataset using both the `translators` library and the ChatGPT API provided by OpenAI. It focuses on translating the DialogSum dataset, but it can be adapted for other datasets as well. The script performs the following tasks:

1. Translates text from one language to another using the `translators` library.
2. Translates text from English to Russian using the ChatGPT API.
3. Creates and manages an SQLite database to store translation data.
4. Uploads a dataset into the database.
5. Retrieves rows from the database and performs translations if needed.
6. Updates the database with translated data.

## Prerequisites

Before running the script, you need to:

1. Create a file named `api_keys.txt` in the same directory as the script.
2. Add your OpenAI API keys, one key per line, to the `api_keys.txt` file.

## Configuration

You can configure various parameters in the script:

- `FROM_LANGUAGE` and `TO_LANGUAGE`: Source and target language codes.
- `MAX_CONCURRENCY`: Maximum concurrency for parallel processing.
- `TS_ENGINE`: Translation engine to use with the `translators` library.
- `DB_FILE`: SQLite database file name.
- `DIALOG_SUMMARIZATION_DATASETS`: List of datasets to use.
- `API_KEYS`: List of API keys read from `api_keys.txt`.

## Running the Script

To run the script, execute it using Python:

```bash
python .\scripts\translate_async.py
```

## Workflow

1. The script initializes the OpenAI API with one of the API keys randomly selected from `API_KEYS`.

2. It checks if the database file specified in `DB_FILE` exists and if the `dialogsum_translation` table exists in the database. If not, it creates the database and uploads the dataset.

3. The script retrieves rows from the database where the `status` is set to `'pending'`. Each row corresponds to a dialog in the dataset.

4. It spawns multiple tasks to translate each row in parallel, respecting the `MAX_CONCURRENCY` limit.

5. For each row, the script performs the following translation steps:

   - Translates `original_dialog_info` using the `translators` library.
   - Translates `log` using the `translators` library.
   - Translates `original_dialog_info` using the ChatGPT API.
   - Translates `log` using the ChatGPT API.
   - Updates the `status` to `'translated'` for the row.

6. The translated data is stored in the database.

7. The script continues to process other rows in parallel until all rows are translated.

## Notes

- The script uses the `translators` library for basic text translation and the ChatGPT API for more complex translations.
- It handles rate limits by retrying the ChatGPT API request if necessary.
- You can adapt this script for other datasets and translation tasks by modifying the `DIALOG_SUMMARIZATION_DATASETS` list and the translation logic.
- Make sure to handle API keys and database files securely and follow OpenAI's usage guidelines.

Feel free to customize and extend this script to suit your specific translation and dataset requirements.
