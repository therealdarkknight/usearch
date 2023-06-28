
#include "external_retreival_testutils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>


int NUM_VECTORS = -1;
char** nodes;

void print_level_histo(int* all_levels, int len) {
    for (int i = 0; i < len; i++) {
        fprintf(stderr, "level%d %d\n", i, all_levels[i]);
    }
}

void prepare_external_index(char* mapped_index, int dim, size_t num_vectors, usearch_metadata_t* metadata) {
    const int NODE_HEAD_BYTES = sizeof(usearch_label_t) + 4 /*sizeof dim */ + 4 /*sizeof level*/;
    const int VECTOR_BYTES = dim * sizeof(float);

    NUM_VECTORS = num_vectors;

    size_t progress = 64; // sizeof file_header_t = 64
    // todo:: this is not quite dim in bytes. it is dim * sizeof(scalar_t) but for some reason punned_dense makes scalar_t = byte_t
    // since I do not quite understand the logic behind storing this on node_t, come back to look at it when/if things break
    int read_dim_bytes = -1;
    int all_levels[20] = {0};
    int level;
    nodes = (char**)calloc(num_vectors, sizeof(char*));
    assert(nodes != NULL);
    for (size_t i = 0; i < num_vectors; i++) {
        int node_bytes = 0;
        char* tape = mapped_index + progress;
        nodes[i] = tape;
        memcpy(&read_dim_bytes, tape + sizeof(usearch_label_t), 4); //+sizeof(label)
        assert(VECTOR_BYTES == read_dim_bytes);
        memcpy(&level, tape + sizeof(usearch_label_t) + 4, 4); //+sizeof(label)+sizeof(dim)
        if ((size_t)level > sizeof(all_levels) / sizeof(int)) {
            fprintf(stderr, "warn: large level %d\n", level);
        } else {
            all_levels[level]++;
        }

        node_bytes += NODE_HEAD_BYTES + metadata->neighbors_base_bytes;
        node_bytes += metadata->neighbors_bytes * level;
        node_bytes += VECTOR_BYTES;
        progress += node_bytes;
    }
    print_level_histo(all_levels, 20);
}

void* node_retriever(unsigned long int id) { return nodes[id]; }
