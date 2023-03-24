#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <chrono>
#include <iostream>
#define INFINITY 200
using namespace std;
using namespace std::chrono;


int Find_min_dist(int loc_dist[], int loc_known[], int loc_n, int ver) {
    int loc_u = -1, loc_v;
    int shortest_dist = INFINITY;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        if (!loc_known[ver * loc_n + loc_v]) {
            if (loc_dist[ver * loc_n + loc_v] < shortest_dist) {
                shortest_dist = loc_dist[ver * loc_n + loc_v];
                loc_u = loc_v;
            }
        }
    }
    return loc_u;
}


void Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
                   int my_rank, int loc_n, int n, int rem) {
    int offset;
    if(my_rank<rem){
        offset=my_rank * loc_n;
    }else{
        offset=(rem * (loc_n+1))+((my_rank-rem)*(loc_n));
    }
    for(int i=0;i<n;i++){
        int loc_v;

        for (loc_v = 0; loc_v < loc_n; loc_v++)
            loc_known[i * loc_n + loc_v] = 0;
        if(i<(offset+loc_n) && i>=offset){
            loc_known[i*loc_n + i-offset]=1;
        }


        for (loc_v = 0; loc_v < loc_n; loc_v++) {
            loc_dist[i * loc_n + loc_v] = loc_mat[i * loc_n + loc_v];
            loc_pred[i * loc_n + loc_v] = 0;
        }
    }
}



void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
              MPI_Comm comm,int rem) {

    int i, loc_v, loc_u, glbl_u, new_dist, my_rank, dist_glbl_u;
    int *loc_known;
    int *my_min = (int *)malloc(n* 2 * sizeof(int));
    int *glbl_min = (int *)malloc(n* 2 * sizeof(int));

    MPI_Comm_rank(comm, &my_rank);
    loc_known = (int *)malloc(n * loc_n * sizeof(int));

    Init(loc_mat, loc_pred, loc_dist, loc_known, my_rank, loc_n, n, rem);

    for(int ver=0;ver<n;ver++){
    for (i = 0; i < n - 1; i++) {
        int offset;
        if(my_rank<rem){
            offset=my_rank * loc_n;
        }else{
            offset=(rem * (loc_n+1))+((my_rank-rem)*(loc_n));
        }

        loc_u = Find_min_dist(loc_dist, loc_known, loc_n,ver);

        if (loc_u != -1) {
            my_min[ver * 2] = loc_dist[ver * loc_n + loc_u];
            my_min[ver * 2 + 1] = loc_u+offset;
        }
        else {
            my_min[ver * 2] = INFINITY;
            my_min[ver * 2 + 1] = -1;
        }

        MPI_Allreduce(my_min+ver * 2, glbl_min+ver * 2, 1, MPI_2INT, MPI_MINLOC, comm);

        dist_glbl_u = glbl_min[ver * 2];
        glbl_u = glbl_min[ver * 2 + 1];

        if (glbl_u == -1)
            break;

    if(glbl_u<(offset+loc_n) && glbl_u>=offset){
            loc_known[ver*loc_n + glbl_u-offset]=1;
        }
 
        for (loc_v = 0; loc_v < loc_n; loc_v++) {
            if (!loc_known[ver * loc_n + loc_v]) {
                new_dist = dist_glbl_u + loc_mat[glbl_u * loc_n + loc_v];
                if (new_dist < loc_dist[ver * loc_n + loc_v]) {
                    loc_dist[ver * loc_n + loc_v] = new_dist;
                    loc_pred[ver * loc_n + loc_v] = glbl_u;
                }
            }
        }
    }
    }
    free(loc_known);
}

int main(int argc, char **argv) {
    int *loc_mat, *loc_dist, *loc_pred, *global_dist = NULL, *global_pred = NULL;
    int my_rank, p, loc_n, n, m;
    MPI_Comm comm;
    MPI_Datatype blk_col_mpi_t;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &p);
    ///////////////////////////////////////
    int *d = NULL;
    auto start= high_resolution_clock::now();
    if (my_rank == 0){
        FILE *infile = fopen(argv[1], "r");
        fscanf(infile, "%d %d", &n, &m);
        d = (int *) malloc(sizeof(int *) * n * n);
        for (int i = 0; i < n * n; ++i) d[i] = INFINITY;
	for (int i=0;i<n;++i) d[i*n+i]=0;
        int a, b, w;
        for (int i = 0; i < m; ++i) {
            fscanf(infile, "%d %d %d", &a, &b, &w);
            d[a * n + b] = d[b * n + a] = w;
        }
        fclose(infile);
	start = high_resolution_clock::now();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    ////////////////////////////////////////
    loc_n = n / p;
    int just=loc_n+1;
    int rem=n%p;
    if (my_rank<rem){
        loc_n = loc_n+1;
    }
    loc_mat = (int *)malloc(n * loc_n * sizeof(int));
    loc_dist = (int *)malloc(n * loc_n * sizeof(int));
    loc_pred = (int *)malloc(n * loc_n * sizeof(int));

    if (my_rank == 0) {
        global_dist = (int *)malloc(n * n * sizeof(int));
        global_pred = (int *)malloc(n * n * sizeof(int));
    }
    ////////////////////////////////////////////////////////
    int * displs = (int *)malloc(p*sizeof(int));
    int * scounts = (int *)malloc(p*sizeof(int));
    int offset = 0;
    for (int i=0; i<p; ++i) {
	displs[i] = offset;
	if(i<rem){
	    scounts[i] = (n / p)+1;
	    offset += scounts[i];
	}else{
	    scounts[i] = (n / p);
            offset += scounts[i];
	}
    }
    offset+=(n/p); 
    for(int i=0;i<n;i++){
        MPI_Scatterv(d+i*n, scounts, displs, MPI_INT, loc_mat+i * loc_n,  loc_n, MPI_INT, 0, comm);
    }

    if (my_rank == 0) free(d);
    ////////////////////////////////////////////////////////
    Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, n, comm,rem);
    for(int i=0;i<n;i++){
        MPI_Gatherv((loc_dist+i * loc_n), loc_n, MPI_INT, (global_dist+i * n), scounts, displs ,MPI_INT, 0, comm);
    }
    if (my_rank == 0) {
	    auto stop = high_resolution_clock::now();
	    auto duration = duration_cast<microseconds>(stop - start);
	    cout << "Time taken by function: "<< duration.count() << " microseconds" << endl;
        //Print_dists(global_dist, n);
	    FILE *outfile = fopen(argv[2], "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(outfile, "%d%s",
                    (i == j ? 0 : global_dist[i * n + j]),
                    (j == n - 1 ? " \n" : " ")
                );
            }
        }
        free(global_dist);
        free(global_pred);
    }
    free(loc_mat);
    free(loc_pred);
    free(loc_dist);
    MPI_Finalize();
    return 0;
}









