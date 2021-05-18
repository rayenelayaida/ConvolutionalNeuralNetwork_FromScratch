#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include<conio.h>
#include<string.h>
#include <ctype.h>
#include "Convolutional_Neural_Network.h"
#define SIZE 224
#define CONV_SIZE 3
int numthreads;



float ***image;
int convolution_shape[13][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }

};

float *****convolution_weight;
float **convolution_biais;
int dense_shape[3][2] = {
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};

float ***dense_weight;
float **dense_biais;


int block_shape[3] = {512, SIZE, SIZE};


float ***block1;
float ***block2;
int block_dense_shape = { 512 * 7 * 7 };
float *block1_dense;
float *block2_dense;



void reset_block(float ***mem) {
	int i, j, k;
	for (i = 0; i < block_shape[0]; i++) {
		for (j = 0; j < block_shape[1]; j++) {
			for (k = 0; k < block_shape[2]; k++) {
				mem[i][j][k] = 0.0;
			}
		}
	}
}

void find_label(FILE *file, int index_max_pred){
	int cpt = 0; 
	char buffeur[1024];
	while (!feof(file)) {
		fgets(buffeur, 1024, file);
		if (strlen(buffeur) == 0) {
			break;
		}
		
		if(index_max_pred == cpt){
			printf("\n Elle est predite : %s\n",buffeur);
		}
		cpt = cpt + 1;
    }
}

void reset_block_dense(float *mem) {
	int i;
	for (i = 0; i < block_dense_shape; i++) {
		mem[i] = 0.0;
	}
}

void init_image(){
	image = malloc(3 * sizeof(float**));
	int i,j;
	for (i = 0; i < 3; i++) {
		image[i] = malloc(SIZE * sizeof(float*));
		for (j = 0; j < SIZE; j++) {
			image[i][j] = malloc(SIZE * sizeof(float));
		}
	}
}

void init_convolution_weight(){
	int i, j, k, l;
	convolution_weight = malloc(13 * sizeof(float****));
	convolution_biais = malloc(13 * sizeof(float*));
	for (l = 0; l < 13; l++) {
		convolution_weight[l] = malloc(convolution_shape[l][0] * sizeof(float***));
		for (i = 0; i < convolution_shape[l][0]; i++) {
			convolution_weight[l][i] = malloc(convolution_shape[l][1] * sizeof(float**));
			for (j = 0; j < convolution_shape[l][1]; j++) {
				convolution_weight[l][i][j] = malloc(convolution_shape[l][2] * sizeof(float*));
				for (k = 0; k < convolution_shape[l][2]; k++) {
					convolution_weight[l][i][j][k] = malloc(convolution_shape[l][3] * sizeof(float));
				}
			}
		}
		convolution_biais[l] = malloc(convolution_shape[l][0] * sizeof(float));
	}

}

void init_dense_weight(){
	int l,i;
	dense_weight = malloc(3 * sizeof(float**));
	dense_biais = malloc(3 * sizeof(float*));
	for (l = 0; l < 3; l++) {
		dense_weight[l] = malloc(dense_shape[l][0] * sizeof(float*));
		for (i = 0; i < dense_shape[l][0]; i++) {
			dense_weight[l][i] = malloc(dense_shape[l][1] * sizeof(float));
		}
		dense_biais[l] = malloc(dense_shape[l][1] * sizeof(float));
	}
}

void init_blocks(){
	int i,j;
	block1 = malloc(block_shape[0] * sizeof(float**));
	block2 = malloc(block_shape[0] * sizeof(float**));
	for (i = 0; i < block_shape[0]; i++) {
		block1[i] = malloc(block_shape[1] * sizeof(float*));
		block2[i] = malloc(block_shape[1] * sizeof(float*));
		for (j = 0; j < block_shape[1]; j++) {
			block1[i][j] = malloc(block_shape[2] * sizeof(float));
			block2[i][j] = malloc(block_shape[2] * sizeof(float));
		}
	}
}

void init_blocks_dense(){
	block1_dense = calloc(block_dense_shape, sizeof(float));
	block2_dense = calloc(block_dense_shape, sizeof(float));
}


void initialization() {
	init_image();
	init_convolution_weight();
	init_dense_weight();
	init_blocks();
	reset_block(block1);
	reset_block(block2);
	init_blocks_dense();
}

void free_image(){
	int i,j;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			free(image[i][j]);
		}
		free(image[i]);
	}
	free(image);
}

void free_convolution_weights(){
	int i, j, k, l;
	for (l = 0; l < 13; l++) {
		for (i = 0; i < convolution_shape[l][0]; i++) {
			for (j = 0; j < convolution_shape[l][1]; j++) {
				for (k = 0; k < convolution_shape[l][2]; k++) {
					free(convolution_weight[l][i][j][k]);
				}
				free(convolution_weight[l][i][j]);
			}
			free(convolution_weight[l][i]);
		}
		free(convolution_weight[l]);
		free(convolution_biais[l]);
	}
	free(convolution_weight);
	free(convolution_biais);
}

void free_dense_weights(){
	int l,i;
	for (l = 0; l < 3; l++) {
		for (i = 0; i < dense_shape[l][0]; i++) {
			free(dense_weight[l][i]);
		}
		free(dense_weight[l]);
		free(dense_biais[l]);
	}
	free(dense_weight);
	free(dense_biais);

}

void free_blocks(){
	int i,j;
	for (i = 0; i < block_shape[0]; i++) {
		for (j = 0; j < block_shape[1]; j++) {
			free(block1[i][j]);
			free(block2[i][j]);
		}
		free(block1[i]);
		free(block2[i]);
	}
	free(block1);
	free(block2);
}

void free_dense_blocks(){
	free(block1_dense);
	free(block2_dense);
}
void free_all() {
	free_image();
	free_convolution_weights();
	free_dense_weights();
	free_blocks();
	free_dense_blocks();
}


void read_weights_file(char *in_file, int levels) {
	float dval;
	int i, j, k, l, z;
	FILE *iin;
	int total_levels_read = 0;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}
	

	for (z = 0; z < 13; z++) {
		if (total_levels_read >= levels && levels != -1)
			break;
		printf("Read conv block %d weights\n", z);
		for (i = 0; i < convolution_shape[z][0]; i++) {
			for (j = 0; j < convolution_shape[z][1]; j++) {
				for (k = 0; k < convolution_shape[z][2]; k++) {
					for (l = 0; l < convolution_shape[z][3]; l++) {
						fscanf(iin, "%f", &dval);
						convolution_weight[z][i][j][CONV_SIZE - k - 1][CONV_SIZE - l - 1] = dval;
					}
				}
			}
		}
		for (i = 0; i < convolution_shape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			convolution_biais[z][i] = dval;
		}
		total_levels_read += 1;
	}

	for (z = 0; z < 3; z++) {
		if (total_levels_read >= levels && levels != -1)
			break;
		printf("Read dense block %d weights\n", z);
		for (i = 0; i < dense_shape[z][0]; i++) {
			for (j = 0; j < dense_shape[z][1]; j++) {
				fscanf(iin, "%f", &dval);
				dense_weight[z][i][j] = dval;
			}
		}
		for (i = 0; i < dense_shape[z][1]; i++) {
			fscanf(iin, "%f", &dval);
			dense_biais[z][i] = dval;
		}
		total_levels_read += 1;
	}

	fclose(iin);
}


void read_image_file(char *in_file) {
	int i, j, l;
	FILE *iin;
	float dval;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			for (l = 0; l < 3; l++) {
				fscanf(iin, "%f", &dval);
				image[l][i][j] = dval;
			}
		}
	}

	fclose(iin);
}


void image_normalization() {
	int i, j, l;
	float coef[3] = { 103.939, 116.779, 123.68 };

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[l][i][j] -= coef[l];
			}
		}
	}
}


void convolution(float **matrix, float **kernel, float **out, int size) {
	int i, j;
	float sum;
	float zeropadding[SIZE + 2][SIZE + 2] = { 0.0 };

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropadding[i + 1][j + 1] = matrix[i][j];
		}
	}

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			sum = zeropadding[i][j] * kernel[0][0] +
				zeropadding[i + 1][j] * kernel[1][0] +
				zeropadding[i + 2][j] * kernel[2][0] +
				zeropadding[i][j + 1] * kernel[0][1] +
				zeropadding[i + 1][j + 1] * kernel[1][1] +
				zeropadding[i + 2][j + 1] * kernel[2][1] +
				zeropadding[i][j + 2] * kernel[0][2] +
				zeropadding[i + 1][j + 2] * kernel[1][2] +
				zeropadding[i + 2][j + 2] * kernel[2][2];
			out[i][j] += sum;
		}
	}
	
}


void bias_and_relu(float **out, float bs, int size) {
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			out[i][j] += bs;
			if (out[i][j] < 0)
				out[i][j] = 0.0;

		}
	}
}


void bias_and_relu_flatten(float *out, float *bs, int size, int relu) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] += bs[i];
		if (relu == 1) {
			if (out[i] < 0)
				out[i] = 0.0;
		}
	}
}


float maxpooling_2x2(float a, float b, float c, float d) {
	if (a >= b && a >= c && a >= d) {
		return a;
	}
	if (b >= c && b >= d) {
		return b;
	}
	if (c >= d) {
		return c;
	}
	return d;
}


void maxpooling(float **out, int size) {
	int i, j;
	for (i = 0; i < size; i+=2) {
		for (j = 0; j < size; j+=2) {
			out[i / 2][j / 2] = maxpooling_2x2(out[i][j], out[i + 1][j], out[i][j + 1], out[i + 1][j + 1]);
		}
	}
}

void flatten(float ***in, float *out, int shape0, int shape1, int shape2) {
	int i, j, k, total = 0;
	for (i = 0; i < shape0; i++) {
		for (j = 0; j < shape1; j++) {
			for (k = 0; k < shape2; k++) {
				out[total] = in[i][j][k];
				total += 1;
			}
		}
	}

}


void dense(float *in, float **weights, float *out, int sh_in, int sh_out) {
	int i, j;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[j][i];
		}
		out[i] = sum;

	}
}


void softmax(float *out, int sh_out) {
	int i;
	float max_val, sum;
	max_val = out[0];
	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}
	sum = 0.0;
	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i] - max_val);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}


int save_dense_to_file(float *mem, int shape0) {
	FILE *out;
	int i;
	int position_pred_class = 0;
	float val_pred_class = 0;
	int cpt = 0;
	out = fopen("result_prediction.txt", "w");
	for (i = 0; i < shape0; i++) {
		fprintf(out, "%.12lf\n", mem[i]);
		if (val_pred_class<mem[i])
			{
				val_pred_class = mem[i];
				position_pred_class = cpt;
			}
		cpt = cpt+1;
	}
	fclose(out);
	printf("\n Le modele a predit que l'image correspond au label numero  : %d d'ImageNet a %.3lf pourcent\n ",position_pred_class,val_pred_class*100 );
	return position_pred_class;

}


int VGG16() {
	int i, j, position_class;
	int level, cur_size;

	reset_block(block1);
	reset_block(block2);
	reset_block_dense(block1_dense);
	reset_block_dense(block2_dense);
	level = 0;
	cur_size = SIZE;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)

	for (i = 0; i < convolution_shape[level][0]; i++) {
		
		for (j = 0; j < convolution_shape[level][1]; j++) {
		
			convolution(image[j], convolution_weight[level][i][j], block1[i], cur_size);
		}
		
		bias_and_relu(block1[i], convolution_biais[level][i], cur_size);
	}
	
	
	level = 1;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block1[j], convolution_weight[level][i][j], block2[i], cur_size);
		}
		bias_and_relu(block2[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block1);
	
	
	#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		maxpooling(block2[i], cur_size);
	}
	cur_size /= 2;
	
	
	level = 2;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block2[j], convolution_weight[level][i][j], block1[i], cur_size);
		}
		bias_and_relu(block1[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block2);

	
	level = 3;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block1[j], convolution_weight[level][i][j], block2[i], cur_size);
		}
		bias_and_relu(block2[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block1);
	
	
	#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		maxpooling(block2[i], cur_size);
	}
	cur_size /= 2;

	
	level = 4;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block2[j], convolution_weight[level][i][j], block1[i], cur_size);
		}
		bias_and_relu(block1[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block2);

	
	level = 5;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block1[j], convolution_weight[level][i][j], block2[i], cur_size);
		}
		bias_and_relu(block2[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block1);

	
	level = 6;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block2[j], convolution_weight[level][i][j], block1[i], cur_size);
		}
		bias_and_relu(block1[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block2);
	
	
	#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		maxpooling(block1[i], cur_size);
	}
	cur_size /= 2;
	
	
	level = 7;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block1[j], convolution_weight[level][i][j], block2[i], cur_size);
		}
		bias_and_relu(block2[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block1);

	
	level = 8;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block2[j], convolution_weight[level][i][j], block1[i], cur_size);
		}
		bias_and_relu(block1[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block2);

	
	level = 9;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block1[j], convolution_weight[level][i][j], block2[i], cur_size);
		}
		bias_and_relu(block2[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block1);
	
	
	#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		maxpooling(block2[i], cur_size);
	}
	cur_size /= 2;
	
	
	level = 10;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block2[j], convolution_weight[level][i][j], block1[i], cur_size);
		}
		bias_and_relu(block1[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block2);

	
	level = 11;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block1[j], convolution_weight[level][i][j], block2[i], cur_size);
		}
		bias_and_relu(block2[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block1);

	
	level = 12;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		for (j = 0; j < convolution_shape[level][1]; j++) {
			convolution(block2[j], convolution_weight[level][i][j], block1[i], cur_size);
		}
		bias_and_relu(block1[i], convolution_biais[level][i], cur_size);
	}
	reset_block(block2);
	
	
	#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < convolution_shape[level][0]; i++) {
		maxpooling(block1[i], cur_size);
	}
	cur_size /= 2;
	
	
	flatten(block1, block1_dense, convolution_shape[level][0], cur_size, cur_size);

	
	level = 0;
	dense(block1_dense, dense_weight[level], block2_dense, dense_shape[level][0], dense_shape[level][1]);
	bias_and_relu_flatten(block2_dense, dense_biais[level], dense_shape[level][1], 1);
	reset_block_dense(block1_dense);

	
	level = 1;
	dense(block2_dense, dense_weight[level], block1_dense, dense_shape[level][0], dense_shape[level][1]);
	bias_and_relu_flatten(block1_dense, dense_biais[level], dense_shape[level][1], 1);
	reset_block_dense(block2_dense);
	
	
	level = 2;
	dense(block1_dense, dense_weight[level], block2_dense, dense_shape[level][0], dense_shape[level][1]);
	bias_and_relu_flatten(block2_dense, dense_biais[level], dense_shape[level][1], 1);
	softmax(block2_dense, dense_shape[level][1]);
	position_class = save_dense_to_file(block2_dense, dense_shape[level][1]);
	
	return position_class;
}


char *trimwhitespace(char *str)
{
	char *end;

	
	while (isspace((unsigned char)*str)) str++;

	if (*str == 0)  
		return str;

	
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end)) end--;

	
	*(end + 1) = 0;

	return str;
}


int main(int argc, char *argv[]) {
	FILE *file_list,*results, *labels;
	char buf[1024];
	int reponse = 1;
	char *weights_file;
	char *image_list_file;
	char *output_file;
	int levels = -1;

#ifdef _OPENMP
	numthreads = omp_get_num_procs() - 1;
#endif
	if (numthreads < 1)
		numthreads = 1;
	
	printf("Nombre de threads :  %d \n", numthreads);

	if (argc != 4) {
		printf("Usage: <program.exe> <weights file> <images list file> <output file>\n");
		return 0;
	}
	weights_file = argv[1];
	image_list_file = argv[2];
	output_file = argv[3];

	initialization();
	read_weights_file(weights_file, levels);
	do{
		file_list = fopen(image_list_file, "r");
		if (file_list == NULL) {
			printf("Fichier inexistant : %s", image_list_file);
			return 1;
		}
		results = fopen(output_file, "w");
		if (results == NULL) {
			printf("Le fichier n'a pas s'ouvrir en ecriture %s", output_file);
			return 1;
		}

		while (!feof(file_list)) {
			fgets(buf, 1024, file_list);
			if (strlen(buf) == 0) {
				break;
			}
			printf("%s\n", buf);
			read_image_file(trimwhitespace(buf));
			image_normalization();
			
			int position_class;
			position_class = VGG16();
			
			labels = fopen("labels.txt","r");
			find_label(labels, position_class);
			
			fclose(labels);  

	   	}

		fclose(file_list);
		reponse = 0;
		printf("\nSi vous voulez predire les classes de d'autres images, veuillez remplir a nouveau le fichier images_list, le sauvegarder puis entrer 1.\n Sinon entrer une autre touche\n");
		scanf("%d",&reponse);
	}while(reponse == 1);
	free_all();
	return 0;
}

