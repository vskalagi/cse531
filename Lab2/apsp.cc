#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#define INFINITY 200
    
int Read_n(int my_rank, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Read_matrix(int loc_mat[], int n, int loc_n, MPI_Datatype blk_col_mpi_t,
                 int my_rank, MPI_Comm comm);
void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
                   int my_rank, int loc_n, int rem);
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
              MPI_Comm comm, int rem);
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n, int ver);
void Print_matrix(int global_mat[], int rows, int cols);
void Print_dists(int global_dist[], int n);
void Print_paths(int global_pred[], int n);

int main(int argc, char **argv) {
    printf("haha\n ");
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
    }
    printf("haha2\n ");
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    ////////////////////////////////////////
    //n = Read_n(my_rank, comm);
    loc_n = n / p;
    int just=loc_n+1;
    int rem=n%p;
    if (my_rank<rem){
        loc_n = loc_n+1;
    }
    loc_mat = (int *)malloc(n * loc_n * sizeof(int));
    loc_dist = (int *)malloc(n * loc_n * sizeof(int));
    loc_pred = (int *)malloc(n * loc_n * sizeof(int));
    blk_col_mpi_t = Build_blk_col_type(n, loc_n);

    if (my_rank == 0) {
        global_dist = (int *)malloc(n * n * sizeof(int));
        global_pred = (int *)malloc(n * n * sizeof(int));
    }
    ////////////////////////////////////////////////////////
    printf("haha3\n ");
    int * displs = (int *)malloc(p*sizeof(int));
    int * scounts = (int *)malloc(p*sizeof(int));
    int offset = 0;
    for (int i=0; i<p; ++i) {
	displs[i] = offset;
	if(i<rem){
	    scounts[i] = (n / p)+1;
	    printf("n/p = %d\n ",scounts[i]);
	    offset += scounts[i];
	}else{
	    scounts[i] = (n / p);
	    printf("--n/p = %d--\n ",scounts[i]);
            offset += scounts[i];
	}
        //offset += stride[i];
        //scounts[i] = 100 - i;
    }
    offset+=(n/p); 
    //MPI_Scatter(d, 1, blk_col_mpi_t, loc_mat, n * loc_n, MPI_INT, 0, comm);
    for(int i=0;i<n;i++){
        MPI_Scatterv(d+i*n, scounts, displs, MPI_INT, loc_mat+i * loc_n,  loc_n, MPI_INT, 0, comm);
    }
    printf("haha4\n ");

    //if (my_rank == 0) free(d);
    if(my_rank==0){
	    for (int i = 0; i < loc_n; ++i) {
            for (int j = 0; j < loc_n; ++j) {

                  printf("-%d ", d[i * n + j]);
            }
        }
	    printf("--now local\n");

	    for (int i = 0; i < loc_n; ++i) {
            for (int j = 0; j < loc_n; ++j) {

                  printf("-%d ", loc_mat[i * loc_n + j]);
            }
        }


    }
    ////////////////////////////////////////////////////////
    Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, n, comm,rem);
    printf("haha5\n ");

    /* Gather the results from Dijkstra */
    /*
    for(int i=0;i<n;i++){
    	MPI_Gather((loc_dist+i * loc_n), loc_n, MPI_INT, (global_dist+i * n), loc_n, MPI_INT, 0, comm);
    	MPI_Gather((loc_pred+i * loc_n), loc_n, MPI_INT, (global_pred+i * n), loc_n, MPI_INT, 0, comm);
    }*/
    for(int i=0;i<n;i++){
        MPI_Gatherv((loc_dist+i * loc_n), loc_n, MPI_INT, (global_dist+i * n), scounts, displs ,MPI_INT, 0, comm);
        //MPI_Gatherv((loc_pred+i * loc_n), loc_n, MPI_INT, (global_pred+i * n), scounts, displs , MPI_INT, 0, comm);
    }
    printf("haha6\n ");

    

    /* Print results */
    if (my_rank == 0) {
        Print_dists(global_dist, n);
	FILE *outfile = fopen(argv[2], "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(outfile, "%d%s",
                    (i == j ? 0 : global_dist[i * n + j]),
                    (j == n - 1 ? " \n" : " ")
                );
            }
        }

	for (int i = 0; i < loc_n; ++i) {
            for (int j = 0; j < loc_n; ++j) {
              
                  printf("-%d ", loc_dist[i * loc_n + j]);
            }
        }

        //Print_paths(global_pred, n);
        free(global_dist);
        free(global_pred);
    }
    printf("haha7\n ");
    free(loc_mat);
    free(loc_pred);
    free(loc_dist);
    MPI_Type_free(&blk_col_mpi_t);
    MPI_Finalize();
    return 0;
}






/*---------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Read in the number of rows in the matrix on process 0
 *            and broadcast this value to the other processes
 * In args:   my_rank:  the calling process' rank
 *            comm:  Communicator containing all calling processes
 * Ret val:   n:  the number of rows in the matrix
 */

/*int Read_n(int my_rank, MPI_Comm comm,char** argv) {
    int n;

    if (my_rank == 0)
        //scanf("%d", &n);
        n=argv[]

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    return n;
}
*/





/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n) {
    MPI_Aint lb, extent;
    MPI_Datatype block_mpi_t;
    MPI_Datatype first_bc_mpi_t;
    MPI_Datatype blk_col_mpi_t;

    MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
    MPI_Type_get_extent(block_mpi_t, &lb, &extent);

    /* MPI_Type_vector(numblocks, elts_per_block, stride, oldtype, *newtype) */
    MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);

    /* This call is needed to get the right extent of the new datatype */
    MPI_Type_create_resized(first_bc_mpi_t, lb, extent, &blk_col_mpi_t);

    MPI_Type_commit(&blk_col_mpi_t);

    MPI_Type_free(&block_mpi_t);
    MPI_Type_free(&first_bc_mpi_t);

    return blk_col_mpi_t;
}






/*---------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in an nxn matrix of ints on process 0, and
 *            distribute it among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 * In args:   n:  the number of rows/cols in the matrix and the submatrices
 *            loc_n = n/p:  the number of columns in the submatrices
 *            blk_col_mpi_t:  the MPI_Datatype used on process 0
 *            my_rank:  the caller's rank in comm
 *            comm:  Communicator consisting of all the processes
 * Out arg:   loc_mat:  the calling process' submatrix (needs to be
 *               allocated by the caller)
 */
void Read_matrix(int loc_mat[], int n, int loc_n,
                 MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
    int *mat = NULL, i, j;

    if (my_rank == 0) {
        mat = (int *)malloc(n * n * sizeof(int));
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                scanf("%d", &mat[i * n + j]);
    }

    MPI_Scatter(mat, 1, blk_col_mpi_t, loc_mat, n * loc_n, MPI_INT, 0, comm);

    if (my_rank == 0) free(mat);
}






/*-------------------------------------------------------------------
 * Function:   Dijkstra_Init
 * Purpose:    Initialize all the matrices so that Dijkstras shortest path
 *             can be run
 *
 * In args:    loc_n:    local number of vertices
 *             my_rank:  the process rank
 *
 * Out args:   loc_mat:  local matrix containing edge costs between vertices
 *             loc_dist: loc_dist[v] = shortest distance from the source to each vertex v
 *             loc_pred: loc_pred[v] = predecessor of v on a shortest path from source to v
 *             loc_known: loc_known[v] = 1 if vertex has been visited, 0 else
 *
 *
 */
void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
                   int my_rank, int loc_n, int n, int rem) {
    printf("haha411\n ");
    int offset;
    if(my_rank<rem){
        offset=my_rank * loc_n;
    }else{
        offset=(rem * (loc_n+1))+((my_rank-rem)*(loc_n));
    }
    printf("haha412\n ");
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
    printf("haha413\n ");
}






/*-------------------------------------------------------------------
 * Function:   Dijkstra
 * Purpose:    compute all the shortest paths from the source vertex 0
 *             to all vertices v
 *
 *
 * In args:    loc_mat:  local matrix containing edge costs between vertices
 *             loc_n:    local number of vertices
 *             n:        total number of vertices (globally)
 *             comm:     the communicator
 *
 * Out args:   loc_dist: loc_dist[v] = shortest distance from the source to each vertex v
 *             loc_pred: loc_pred[v] = predecessor of v on a shortest path from source to v
 *
 */
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
              MPI_Comm comm,int rem) {
    printf("haha41\n ");

    int i, loc_v, loc_u, glbl_u, new_dist, my_rank, dist_glbl_u;
    int *loc_known;
    int *my_min = (int *)malloc(n* 2 * sizeof(int));//int my_min[2];
    int *glbl_min = (int *)malloc(n* 2 * sizeof(int));//int glbl_min[2];

    MPI_Comm_rank(comm, &my_rank);
    loc_known = (int *)malloc(n * loc_n * sizeof(int));

    Dijkstra_Init(loc_mat, loc_pred, loc_dist, loc_known, my_rank, loc_n, n, rem);
    printf("haha42\n ");

    /* Run loop n - 1 times since we already know the shortest path to global
       vertex 0 from global vertex 0 */
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
            my_min[ver * 2 + 1] = loc_u+offset;//loc_u + my_rank * loc_n;
        }
        else {
            my_min[ver * 2] = INFINITY;
            my_min[ver * 2 + 1] = -1;
        }

        /* Get the minimum distance found by the processes and store that
           distance and the global vertex in glbl_min
        */
        MPI_Allreduce(my_min+ver * 2, glbl_min+ver * 2, 1, MPI_2INT, MPI_MINLOC, comm);

        dist_glbl_u = glbl_min[ver * 2];
        glbl_u = glbl_min[ver * 2 + 1];

        /* This test is to assure that loc_known is not accessed with -1 */
        if (glbl_u == -1)
            break;

        /* Check if global u belongs to process, and if so update loc_known */
        /*if ((glbl_u / loc_n) == my_rank) {
            loc_u = glbl_u % loc_n;
            loc_known[ver * loc_n + loc_u] = 1;
        }*/
	//int offset;
        //if(my_rank<rem){
        //    offset=my_rank * loc_n;
        //}else{
        //    offset=(rem * (loc_n+1))+((my_rank-rem)*(loc_n));
        //}
	//if(glbl_u>offset && glbl_u<)
	if(glbl_u<(offset+loc_n) && glbl_u>=offset){
            loc_known[ver*loc_n + glbl_u-offset]=1;
        }
 

        /* For each local vertex (global vertex = loc_v + my_rank * loc_n)
           Update the distances from source vertex (0) to loc_v. If vertex
           is unmarked check if the distance from source to the global u + the
           distance from global u to local v is smaller than the distance
           from the source to local v
         */
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






/*-------------------------------------------------------------------
 * Function:   Find_min_dist
 * Purpose:    find the minimum local distance from the source to the
 *             assigned vertices of the process that calls the method
 *
 *
 * In args:    loc_dist:  array with distances from source 0
 *             loc_known: array with values 1 if the vertex has been visited
 *                        0 if not
 *             loc_n:     local number of vertices
 *
 * Return val: loc_u: the vertex with the smallest value in loc_dist,
 *                    -1 if all vertices are already known
 *
 * Note:       loc_u = -1 is not supposed to be used when this function returns
 *
 */
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






/*-------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print the contents of the matrix
 * In args:   mat, rows, cols
 *
 *
 */
void Print_matrix(int mat[], int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            if (mat[i * cols + j] == INFINITY)
                printf("i ");
            else
                printf("%d ", mat[i * cols + j]);
        printf("\n");
    }

    printf("\n");
}






/*-------------------------------------------------------------------
 * Function:    Print_dists
 * Purpose:     Print the length of the shortest path from 0 to each
 *              vertex
 * In args:     n:  the number of vertices
 *              dist:  distances from 0 to each vertex v:  dist[v]
 *                 is the length of the shortest path 0->v
 */
void Print_dists(int global_dist[], int n) {
    int v;

    printf("  v    dist 0->v\n");
    printf("----   ---------\n");

    for (v = 98*n; v < 99*n; v++) {
        if (global_dist[v] == INFINITY) {
            printf("%3d       %5s\n", v, "inf");
        }
        else
            printf("%3d       %4d\n", v, global_dist[v]);
        }
    for (v = 99*n; v < 100 * n; v++) {
        if (global_dist[v] == INFINITY) {
            printf("%3d       %5s\n", v, "inf");
        }
        else
            printf("%3d       %4d\n", v, global_dist[v]);
        }

    printf("\n");
}






/*-------------------------------------------------------------------
 * Function:    Print_paths
 * Purpose:     Print the shortest path from 0 to each vertex
 * In args:     n:  the number of vertices
 *              pred:  list of predecessors:  pred[v] = u if
 *                 u precedes v on the shortest path 0->v
 */
void Print_paths(int global_pred[], int n) {
    int v, w, *path, count, i;

    path =  (int *)malloc(n * sizeof(int));

    printf("  v     Path 0->v\n");
    printf("----    ---------\n");
    for (v = 1; v < n; v++) {
        printf("%3d:    ", v);
        count = 0;
        w = v;
        while (w != 0) {
            path[count] = w;
            count++;
            w = global_pred[w];
        }
        printf("0 ");
        for (i = count-1; i >= 0; i--)
            printf("%d ", path[i]);
        printf("\n");
    }

    free(path);
}





