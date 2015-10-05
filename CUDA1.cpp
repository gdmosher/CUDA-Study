float *h_dataA, *h_dataB, *h_resultC;
float *d_dataA, *d_dataB, *d_resultC;

h_dataA     = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
h_dataB     = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
h_resultC = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
CUDA_SAFE_CALL( cudaMalloc( (void **)&d_dataA, sizeof(float) * MAX_DATA_SIZE) );
CUDA_SAFE_CALL( cudaMalloc( (void **)&d_dataB, sizeof(float) * MAX_DATA_SIZE) );
CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resultC , sizeof(float) * MAX_DATA_SIZE) );