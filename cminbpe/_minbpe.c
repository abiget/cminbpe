#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include "khash.h"

KHASH_MAP_INIT_INT64(pair, uint64_t)  // pair_key -> frequency count
KHASH_MAP_INIT_INT64(merge, uint64_t) // pair_key -> vocab_id assigned to that merge

typedef struct
{
    /**
     * A structure representing a chunk of tokens.
     * It contains an array of uint32_t tokens and its length.
     */
    uint32_t *tokens;
    size_t len;
} Chunk;

typedef struct
{
    /**
     * A structure representing a list of chunk indices.
     * It contains an array of size_t indices, its length, and its allocated size.
     * This is used to keep track of which chunks contain a specific pair of tokens.
     */
    size_t *indices;
    size_t len;
    size_t size;
} ChunkIndexList;

KHASH_MAP_INIT_INT64(chunkindices, ChunkIndexList *) // pair_key -> ChunkIndexList*
KHASH_SET_INIT_INT64(seen_set)                       // set of chunk indices, reused per-iteration to avoid double-processing a chunk that appears more than once in a pair's chunklist

static PyObject *pack_merge_maps_to_dict(khash_t(merge) * merge_maps);

static int store_chunk_from_bytes(PyObject *chunk, Chunk *out)
{
    /**
     * Convert a Python bytes object to a Chunk structure.
     * The Chunk structure contains an array of uint32_t tokens and its length.
     * This function allocates memory for the tokens array, which must be freed later.
     *
     * @param chunk: A Python bytes object representing the chunk.
     * @param out: A pointer to a Chunk structure where the result will be stored.
     * @return: 0 on success, -1 on failure (with an appropriate Python exception set).
     */
    if (!PyBytes_Check(chunk))
    {
        PyErr_SetString(PyExc_TypeError, "chunks must contain bytes");
        return -1;
    }

    Py_ssize_t len = PyBytes_Size(chunk);

    if (len == 0)
    {
        out->tokens = NULL;
        out->len = 0;
        return 0;
    }

    // Allocate memory for the tokens
    uint32_t *tokens = malloc((size_t)len * sizeof(uint32_t));
    if (tokens == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }

    // Copy the bytes into the tokens array: why casting to unsigned char? Because PyBytes_AsString returns a char*, but we want to treat the bytes as unsigned values (0-255) when converting to uint32_t. This ensures that we correctly handle byte values above 127, which would otherwise be interpreted as negative values if treated as signed chars.
    const unsigned char *data = (const unsigned char *)PyBytes_AsString(chunk);

    for (Py_ssize_t i = 0; i < len; i++)
    {
        tokens[i] = data[i];
    }
    out->tokens = tokens;
    out->len = (size_t)len;
    return 0;
}

static void free_chunks(Chunk *chunks, Py_ssize_t n)
{
    /**
     * Free the memory allocated for an array of Chunk structures.
     *
     * @param chunks: An array of Chunk structures to free.
     * @param n: The number of Chunk structures already allocated in memory.
     */
    if (chunks == NULL)
        return;

    for (Py_ssize_t i = 0; i < n; i++)
    {
        free(chunks[i].tokens);
    }
    free(chunks);
    // set the pointer to NULL to avoid dangling pointer issues
    chunks = NULL;
}

static uint64_t make_unique_key(uint32_t a, uint32_t b)
{
    /**
     * Create a unique key representing a pair of tokens.
     *
     * @param a: The first token.
     * @param b: The second token.
     * @return: A unique uint64_t key representing the pair.
     */

    return ((uint64_t)a << 32) | (uint64_t)b;
}

static int update_pair_count(khash_t(pair) * pair_maps, uint64_t key, int64_t delta)
{
    /**
     * Update the frequency count for a given pair key.
     *
     * @param pair_maps: A hash map that maps pair keys to their frequency counts.
     * @param key: The unique key representing the pair.
     * @param delta: The amount to add to the frequency count (can be negative).
     * @return: 0 on success, -1 on failure.
     */
    khiter_t k = kh_get(pair, pair_maps, key);
    if (k == kh_end(pair_maps))
    {
        int ret;
        k = kh_put(pair, pair_maps, key, &ret);
        if (ret < 0)
            return -1;
        kh_val(pair_maps, k) = (uint64_t)delta;
        return 0;
    }
    int64_t new_val = (int64_t)kh_val(pair_maps, k) + delta;
    if (new_val <= 0)
        kh_del(pair, pair_maps, k);
    else
        kh_val(pair_maps, k) = (uint64_t)new_val;
    return 0;
}

static void free_key_chunk_indices(ChunkIndexList *list)
{
    /**
     * Free the memory allocated for a ChunkIndexList structure.
     *
     * @param list: A pointer to the ChunkIndexList to free.
     */
    if (list == NULL)
        return;
    free(list->indices);
    free(list);
}

static ChunkIndexList *allocate_key_chunk_indices(size_t initial_capacity)
{
    /**
     * Allocate a new ChunkIndexList structure for each key (pair of tokens).
     * This structure is used to keep track of which chunks contain a specific pair of tokens.
     * @param initial_capacity: The initial capacity for the indices array in the ChunkIndexList.
     *
     * @return: A pointer to the newly allocated ChunkIndexList, or NULL on failure.
     */

    ChunkIndexList *list = malloc(sizeof(ChunkIndexList));
    if (list == NULL)
        return NULL;

    list->len = 0;
    list->size = initial_capacity;
    list->indices = malloc(initial_capacity * sizeof(size_t));

    if (list->indices == NULL)
    {
        free(list);
        return NULL;
    }

    return list;
}

static int push_chunk_index(ChunkIndexList *list, size_t idx)
{
    /**
     * Add a chunk index to the ChunkIndexList for a specific pair key.
     * This function dynamically resizes the indices array as needed.
     *
     * @param list: A pointer to the ChunkIndexList to update.
     * @param idx: The chunk index to add.
     * @return: 0 on success, -1 on failure (e.g., memory allocation failure).
     */

    if (list->len == list->size)
    {
        // Resize the indices array if it's full by doubling its capacity.
        int new_capacity = list->size * 2;

        size_t *si = realloc(list->indices, new_capacity * sizeof(size_t));

        if (si == NULL)
            return -1;
        list->indices = si;
        list->size = new_capacity;
    }

    list->indices[list->len++] = idx;

    return 0;
}

static int register_chunks_for_pair(khash_t(chunkindices) * chunkindices_map, uint64_t key, size_t chunk_idx, size_t initial_capacity)
{
    /**
     * Register a chunk index for a given pair key in the pair-to-chunks mapping.
     *
     * @param chunkindices_map: A hash map that maps pair keys to lists of chunk indices where they occur.
     * @param key: The unique key representing the pair.
     * @param chunk_idx: The index of the current chunk to register.
     * @param initial_capacity: The initial capacity for indices in the ChunkIndexList for this pair key.
     * @return: 0 on success, -1 on failure.
     */

    int ret;
    khiter_t k = kh_get(chunkindices, chunkindices_map, key);
    ChunkIndexList *list;

    if (k == kh_end(chunkindices_map))
    {
        // allocate if the key doesn't exist yet
        list = allocate_key_chunk_indices(initial_capacity);
        if (list == NULL)
            return -1;

        // allocate memory for each key
        k = kh_put(chunkindices, chunkindices_map, key, &ret);
        if (ret < 0)
        {
            free_key_chunk_indices(list);
            return -1;
        }

        kh_val(chunkindices_map, k) = list;
    }
    else
    {
        list = kh_val(chunkindices_map, k);
    }

    return push_chunk_index(list, chunk_idx);
}

static int process_chunk_pairs(Chunk *chunk, khash_t(pair) * pair_maps, khash_t(chunkindices) * pair_chunks_map, size_t chunk_idx, int64_t delta, bool register_chunk, size_t initial_capacity)
{
    /**
     * Scan a Chunk for adjacent token pairs and update the pair frequency map and the pair-to-chunks-index mapping.
     *
     * @param chunk: A pointer to the Chunk structure to scan.
     * @param pair_maps: A hash map that maps pair keys to their frequency counts.
     * @param pair_chunks_map: A hash map that maps pair keys to lists of chunk indices where they occur.
     * @param chunk_idx: The index of the current chunk in the overall array of chunks.
     * @param delta: The amount to add to the frequency count for each pair found (can be negative to reverse the contribution of each chunk).
     * @param register_chunk: If true, register the current chunk index in the pair-to-chunks mapping for each pair found.
     * @param initial_capacity: The initial capacity for indices in the ChunkIndexList for each pair key.
     * @return: 0 on success, -1 on failure (with an appropriate Python exception set).
     */

    for (size_t j = 0; j + 1 < chunk->len; j++)
    {
        uint32_t left = chunk->tokens[j];
        uint32_t right = chunk->tokens[j + 1];
        uint64_t key = make_unique_key(left, right);

        // update the frequency count for this pair
        if (update_pair_count(pair_maps, key, delta) < 0)
        {
            return -1;
        }

        if (register_chunk)
        {
            if (register_chunks_for_pair(pair_chunks_map, key, chunk_idx, initial_capacity) < 0)
            {
                return -1;
            }
        }
    }
    return 0;
}
static bool is_chunk_containing_pair(Chunk *chunk, uint32_t left, uint32_t right)
{
    /**
     * Check if a Chunk contains a specific pair of tokens.
     *
     * @param chunk: A pointer to the Chunk structure to check.
     * @param left: The first token of the pair.
     * @param right: The second token of the pair.
     * @return: true if the pair is found in the chunk, false otherwise.
     */
    for (size_t j = 0; j + 1 < chunk->len; j++)
    {
        if (chunk->tokens[j] == left && chunk->tokens[j + 1] == right)
        {
            return true;
        }
    }
    return false;
}

static void merge_pair_in_chunk(Chunk *chunk, uint32_t left, uint32_t right, uint32_t vocab_id)
{
    /**
     * Merge a specific pair of tokens in a Chunk into a new token with the given vocab_id.
     *
     * @param chunk: A pointer to the Chunk structure to modify.
     * @param left: The first token of the pair to merge.
     * @param right: The second token of the pair to merge.
     * @param vocab_id: The new token ID to replace the merged pair.
     * This function modifies the chunk in place, replacing occurrences of the pair (left, right) with vocab_id.
     */
    size_t new_len = 0;
    size_t read = 0;
    while (read < chunk->len)
    {
        if (read + 1 < chunk->len && chunk->tokens[read] == left && chunk->tokens[read + 1] == right)
        {
            // Merge the pair into vocab_id
            chunk->tokens[new_len++] = vocab_id;
            read += 2; // Skip the next token since it's part of the merged pair
        }
        else
        {
            // Keep the current token
            chunk->tokens[new_len++] = chunk->tokens[read];
            read++;
        }
    }
    chunk->len = new_len;
}

static void free_pair_chunk_indices_map(khash_t(chunkindices) * pair_chunks_map)
{
    /**
     * Free the memory allocated for the pair-to-chunks-index mapping.
     *
     * @param pair_chunks_map: A hash map that maps pair keys to lists of chunk indices where they occur.
     */
    if (pair_chunks_map == NULL)
        return;

    for (khiter_t k = kh_begin(pair_chunks_map); k != kh_end(pair_chunks_map); ++k)
    {
        if (kh_exist(pair_chunks_map, k))
        {
            free_key_chunk_indices(kh_val(pair_chunks_map, k));
        }
    }
    kh_destroy(chunkindices, pair_chunks_map);

    // set the pointer to NULL to avoid dangling pointer issues
    pair_chunks_map = NULL;
}

static int bpe_train_loop(
    Chunk *token_chunks, Py_ssize_t num_chunks, khash_t(pair) * pair_maps, khash_t(chunkindices) * pair_chunks_map,
    khash_t(merge) * merge_maps, khash_t(seen_set) * seen_chunks_set, uint64_t vocab_size, uint32_t next_vocab, bool verbose)
{
    /**
     * The main training loop for the BPE algorithm.
     * This function iteratively finds the most frequent pair of tokens and merges them into a new token until the desired vocabulary size is reached.
     *
     * @param token_chunks: An array of Chunk structures representing the input data.
     * @param num_chunks: The number of chunks in the token_chunks array.
     * @param pair_maps: A hash map that maps pair keys to their frequency counts.
     * @param pair_chunks_map: A hash map that maps pair keys to lists of chunk indices where they occur.
     * @param merge_maps: A hash map that maps merged pair keys to their assigned vocabulary IDs.
     * @param seen_chunks_set: A set used to track which chunks have already been processed for a given pair during each iteration.
     * @param vocab_size: The desired size of the vocabulary to be generated.
     * @param next_vocab: The next available vocabulary ID.
     * @param verbose: If true, print progress information during training.
     */

    // The rest of the BPE training logic would go here, including the main loop for merging pairs and updating the vocabulary.
    for (; next_vocab < vocab_size; next_vocab++)
    {
        // The main BPE merge loop would be implemented here,
        // which would involve selecting the most frequent pair, merging it, and updating the relevant data structures.
        // traverse the pair_maps to find the most frequent pair
        uint32_t vocab_id = next_vocab;
        uint64_t best_count = 0;
        uint64_t best_key = 0;

        bool found_best = false;

        for (khiter_t k = kh_begin(pair_maps); k != kh_end(pair_maps); ++k)
        {
            // Process each pair in the map
            if (!kh_exist(pair_maps, k))
                continue;

            uint64_t count = kh_val(pair_maps, k);
            // make sure at least one count is best
            if (!found_best || count > best_count)
            {
                best_count = count;
                best_key = kh_key(pair_maps, k);
                found_best = true;
            }
        }

        if (!found_best)
            break;

        uint32_t best_left = (uint32_t)(best_key >> 32);
        uint32_t best_right = (uint32_t)(best_key & 0xffffffffu);

        // put the best pair into the merge_maps with the new vocab_id
        int ret;
        khiter_t mk = kh_put(merge, merge_maps, best_key, &ret);
        if (ret < 0)
            goto error_terminate;

        kh_val(merge_maps, mk) = vocab_id;

        // why the print show two pairs merged multiple times? Because the same pair can appear in multiple chunks, and each chunk is processed separately. The verbose output is printed for each iteration of the main BPE loop, showing the best pair being merged and its count. If the same pair appears in multiple chunks, it will be counted multiple times, leading to multiple print statements for the same pair merge.
        if (verbose)
        {
            fprintf(stderr,
                    "iteration %u: merging pair (%u, %u) -> vocab_id %u  (count=%llu)\n",
                    next_vocab - 256, best_left, best_right, vocab_id, best_count);
        }
        // now update all chunks that contain this pair, merging them and updating the pair_maps and pair_chunks_map accordingly
        khiter_t clk = kh_get(chunkindices, pair_chunks_map, best_key);

        ChunkIndexList *chunk_indices = kh_end(pair_chunks_map) != clk ? kh_val(pair_chunks_map, clk) : NULL;

        if (chunk_indices == NULL)
            continue;

        for (size_t i = 0; i < chunk_indices->len; i++)
        {
            size_t chunk_idx = chunk_indices->indices[i];

            // check if this chunk has already been processed for this pair in this iteration
            khiter_t sk = kh_get(seen_set, seen_chunks_set, (uint64_t)chunk_idx);
            if (sk != kh_end(seen_chunks_set))
                continue;

            // mark this chunk as processed
            kh_put(seen_set, seen_chunks_set, (uint64_t)chunk_idx, &ret);

            if (ret < 0)
                goto error_terminate;

            Chunk *chunk = &token_chunks[chunk_idx];

            // check if the chunk contains the best pair and merge it
            if (!is_chunk_containing_pair(chunk, best_left, best_right))
                continue;

            // remove the chunk's pair frequencies from the pair_maps before merging
            if (process_chunk_pairs(chunk, pair_maps, pair_chunks_map, chunk_idx, -1, false, 0) < 0)
                goto error_terminate;

            // merge the pair in the chunk
            merge_pair_in_chunk(chunk, best_left, best_right, vocab_id);

            // add the chunk's new pair frequencies to the pair_maps after merging
            if (process_chunk_pairs(chunk, pair_maps, pair_chunks_map, chunk_idx, 1, true, 4) < 0)
                goto error_terminate;
        }

        // reset the seen_chunks_set for the next iteration
        kh_clear(seen_set, seen_chunks_set);

        // best key's count has been decreased to zero by subtracting the contributions of all chunks that contained it,
        // so we can remove it from the pair_maps and free its chunk_indices list
        clk = kh_get(chunkindices, pair_chunks_map, best_key);

        if (clk != kh_end(pair_chunks_map))
        {
            // free the chunk_indices list for this pair key(avoid memory leak if we just remove the key from the pair_chunks_map)
            free_key_chunk_indices(kh_val(pair_chunks_map, clk));
            // remove the pair key from the pair_chunks_map
            kh_del(chunkindices, pair_chunks_map, clk);
        }
    }

    return 0;

error_terminate:
    return -1;
}

static PyObject *train_bpe(PyObject *self, PyObject *args)
{
    /**
     * chunks: list of bytes, the input data to train on e.g. [b'hello', b'world']
     * vocab_size: int, the size of the vocabulary to be generated
     * verbose: int, if 1, print progress information
     */
    PyObject *chunks;
    int vocab_size_arg;
    int verbose = 0;
    int next_vocab = 256;

    if (!PyArg_ParseTuple(args, "Oiii", &chunks, &vocab_size_arg, &verbose, &next_vocab))
        return NULL;

    uint64_t vocab_size = vocab_size_arg;

    if (!PyList_Check(chunks))
    {
        PyErr_SetString(PyExc_TypeError, "chunks must be a list");
        return NULL;
    }
    Py_ssize_t num_chunks = PyList_Size(chunks);

    Chunk *token_chunks = calloc((size_t)num_chunks, sizeof(Chunk));
    if (token_chunks == NULL)
        return PyErr_NoMemory();

    // convert each bytes object in the list to a Chunk
    for (Py_ssize_t i = 0; i < num_chunks; i++)
    {
        PyObject *chunk_bytes = PyList_GET_ITEM(chunks, i);
        // Convert chunk_bytes to a Chunk and store it in token_chunks[i]
        if (store_chunk_from_bytes(chunk_bytes, &token_chunks[i]) < 0)
        {
            free_chunks(token_chunks, i);
            return NULL;
        }
    }

    // initialize hash maps for pair frequencies, pair to chunks mapping, merge mapping, and seen chunks
    khash_t(pair) *pair_maps = kh_init(pair);
    khash_t(chunkindices) *pair_chunks_map = kh_init(chunkindices);
    khash_t(merge) *merge_maps = kh_init(merge);
    khash_t(seen_set) *seen_chunks_set = kh_init(seen_set);

    // check for memory allocation failures
    if (pair_maps == NULL || pair_chunks_map == NULL || merge_maps == NULL || seen_chunks_set == NULL)
        goto nomem_init;

    size_t indice_list_capacity = 4;

    // pre-count pairs in each chunk and populate the pair_maps and pair_chunks_map
    for (Py_ssize_t i = 0; i < num_chunks; i++)
    {
        if (process_chunk_pairs(&token_chunks[i], pair_maps, pair_chunks_map, (size_t)i, 1, true, indice_list_capacity) < 0)
            goto nomem_init;
    }
    size_t next_vocab_id = (size_t)next_vocab;

    if (bpe_train_loop(token_chunks, num_chunks, pair_maps, pair_chunks_map, merge_maps, seen_chunks_set, vocab_size, next_vocab_id, verbose) < 0)
        goto nomem_init;

    // remove all allocated memory for chunks and hash maps except for the merge_maps, which contains the final vocabulary mapping
    free_chunks(token_chunks, num_chunks);
    free_pair_chunk_indices_map(pair_chunks_map);
    kh_destroy(pair, pair_maps);
    kh_destroy(seen_set, seen_chunks_set);

    // convert the merge_maps to a Python dictionary to return
    PyObject *merge_dict = pack_merge_maps_to_dict(merge_maps);
    if (merge_dict == NULL)
    {
        return PyErr_NoMemory();
    }

    kh_destroy(merge, merge_maps);

    return merge_dict;

nomem_init:
    free_chunks(token_chunks, num_chunks);
    if (pair_chunks_map)
        free_pair_chunk_indices_map(pair_chunks_map);
    if (pair_maps)
        kh_destroy(pair, pair_maps);
    if (merge_maps)
        kh_destroy(merge, merge_maps);
    if (seen_chunks_set)
        kh_destroy(seen_set, seen_chunks_set);
    return PyErr_NoMemory();
}

static PyObject *pack_merge_maps_to_dict(khash_t(merge) * merge_maps)
{
    /**
     * Convert the merge_maps hash map to a Python dictionary.
     *
     * @param merge_maps: A hash map that maps merged pair keys to their assigned vocabulary IDs.
     * @return: A Python dictionary representing the merge mappings, or NULL on failure (with an appropriate Python exception set).
     */

    PyObject *result = PyDict_New();
    if (result == NULL)
        return NULL;

    for (khiter_t k = kh_begin(merge_maps); k != kh_end(merge_maps); ++k)
    {
        if (!kh_exist(merge_maps, k))
            continue;

        uint64_t pair_key = kh_key(merge_maps, k);
        uint32_t left = (uint32_t)(pair_key >> 32);
        uint32_t right = (uint32_t)(pair_key & 0xffffffffu);
        uint32_t vocab_id = (uint32_t)kh_val(merge_maps, k);

        PyObject *pair_tuple = Py_BuildValue("(II)", left, right);
        if (pair_tuple == NULL)
        {
            Py_DECREF(result);
            return NULL;
        }

        PyObject *val = PyLong_FromUnsignedLong(vocab_id);
        if (val == NULL)
        {
            Py_DECREF(pair_tuple);
            Py_DECREF(result);
            return NULL;
        }

        if (PyDict_SetItem(result, pair_tuple, val) < 0)
        {
            Py_DECREF(pair_tuple);
            Py_DECREF(val);
            Py_DECREF(result);
            return NULL;
        }

        Py_DECREF(pair_tuple);
        Py_DECREF(val);
    }

    return result;
}

static PyMethodDef MinBPEMethods[] = {
    {"train", train_bpe, METH_VARARGS, "Train BPE on a list of byte chunks."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef minbpemodule = {
    PyModuleDef_HEAD_INIT,
    "_minbpe",
    NULL,
    -1,
    MinBPEMethods,
};

PyMODINIT_FUNC PyInit__minbpe(void)
{
    return PyModule_Create(&minbpemodule);
}