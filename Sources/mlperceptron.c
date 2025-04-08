#include "m_pd.h" // Importa as funções prontas do Pure Data
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
// #include <stdbool.h>





/* Esse código implementa uma rede neural (multilayer perceptron)
O objeto recebe uma lista com nº de camadas, dimensão dos dados de entrada, nº de neurônios em cada camada.
Cria várias matrizes e vetores para representar os pesos, vetor de bias, vetor de ativação, vetor Z e vetor de delta com base nos parâmetros da rede
(aloca, desaloca memória e redimensiona as matrizes e vetores). Recebe lista com dados de entrada e classes para treinamento, float para taxa de aprendizagem,
inteiro para nº de épocas e para modo de treinamento (0 para OFF e 1 para ON). Salva o modelo treinado em um arquivo .txt e carrega o modelo treinado a partir do arquivo .txt.
inicia os pesos e bias com vários métodos de inicialização (aleatório, zeros, uniforme, he, xavier, lecun, ou com intervalos específicos).
Permite a escolha de funções de ativação para cada camada (sigmoid, relu, tanh, prelu, softmax, softplus).
está funcionando sem erros em 2/01/2025
 */


// Define uma nova classe
static t_class *mlperceptron_class;

typedef struct _mlperceptron {
    t_object  x_obj;

    t_int num_hidden; // Número de hidden layers (número de matrizes)
    t_int num_vetores; //número de vetores
    t_int max_epochs; //número máximo de épocas
    t_int current_epoch; //época atual
    t_int datasize; //nº de exemplos de treinamento
    t_int current_data; //exemplo atual
    t_float learn_rate; //taxa de aprendizado
    t_int param; //número de parâmetros recebidos
    t_float trainingmode; //modo de treinamento
    t_float evalmode; //modo de avaliação do modelo treinado
    t_float num_erro; //quantidade de vezes que a rede errou no modo de avaliação
    t_float error; //soma dos erros para cada época
    double cross_entropy;



    t_float *net_param; //ponteiro para o array com os parâmetros da rede
    t_float *class_erro; //ponteiro para o array com número de erros de cada classe
    double *input_data; //ponteiro para o array com os dados de entrada
    double *classes; //ponteiro para o array com as classes
    
    t_symbol **activation_function; //array com nomes de funções de ativação

    double ***hidden_layers; //array de matrizes (número de matrizes, número de linhas e número de colunas)
    // t_float **matriz;  // Variável para armazenar uma matriz alocada dinamicamente (número de linhas e número de colunas)
    double **vetor_bias; //vetor de bias para cada camada
    double **vetor_a; //vetor da soma ponderada de cada camada
    double **vetor_z; //vetor de ativação de cada camada
    double **delta;

    t_atom *x_layer_out; //buffer de saída para enviar as matrizes para o outlet 1
    t_atom *x_linha_out; //buffer de saída para enviar as matrizes para o outlet 2
    t_atom *x_mse; //buffer de saída para enviar o erro médio quadrático para o outlet 2
    t_atom *x_bias_out; //buffer de saída para enviar o vetor de bias para o outlet 2

    t_canvas *x_canvas; //canvas para salvar os parâmetros da rede

    t_outlet  *x_out1; // Outlet 1
    t_outlet  *x_out2; // Outlet 2
} t_mlperceptron;


//-------------------   Função para gerar um número aleatório em um intervalo -------------------------
double random_double(double lower, double upper) {
    return lower + ((double) rand() / RAND_MAX) * (upper - lower);
}


//----------------------- ativa e desativa o modo de treinamento -----------------------
static void training (t_mlperceptron *x, t_floatarg tra){
    // Verifica se o valor recebido é válido (0 ou 1)
    if (tra != 0 && tra != 1) {
        error("Invalid value for training mode. Use 0 for OFF and 1 for ON.");
        return;
    }

    //atualiza o estado de treinamento
    x->trainingmode = tra;
    int training_mode = x->trainingmode;

    //mensagem para indicar o estado do modo de treinamento
    switch(training_mode) {
        case 0:
            post ("training mode: OFF");
            break;
        case 1: 
            x->evalmode = 0;
            post ("training mode: ON");
            break;
    }
}

//------------------------- ativa e desativa o modo de avaliação --------------------//
static void eval_mode (t_mlperceptron *x, t_floatarg eval){
    // Verifica se o valor recebido é válido (0 ou 1)
    if (eval != 0 && eval != 1) {
        error("Invalid value for evaluation mode. Use 0 for OFF and 1 for ON.");
        return;
    }

    //atualiza o estado de treinamento
    x->evalmode = eval;
    int eval_mode = x->evalmode;

    //mensagem para indicar o estado do modo de treinamento
    switch(eval_mode) {
        case 0:
            post ("evaluation mode: OFF");
            break;
        case 1:
            x->num_erro = 0; //reinicia o contador de erros
            x->current_data = 0;
            x->trainingmode = 0;
            for(int c = 0; c < x->net_param[x->param-1]; c++){ //inicia todos valores como 0
                x->class_erro[c] = 0;
            } 
            post ("evaluation mode: ON");
            break;
    }
}

//----------------------------- número de épocas -----------------------------//
static void epoch_amount (t_mlperceptron *x, t_floatarg ep){
    if(ep >=1 ){
        x->max_epochs = (int)ep;
        post("epochs: %d", x->max_epochs);
    } else {
        error("Amount of epochs must be greater than 0.");
    }
}



//----------------------------- número de exemplos de treinamento -----------------------------//
static void datasize (t_mlperceptron *x, t_floatarg data){
    if(data >=1 ){
        x->datasize = (int)data;
        post("Amount of training examples: %d", x->datasize);
    } else {
        error("Amount of training example must be greater than 0.");
    }
}


//------------------------- taxa de aprendizado ------------------------------//
static void learning_rate(t_mlperceptron *x, t_float le) {
    // Verifica se a taxa de aprendizado está no intervalo válido [0, 1]
    if (le >= 0 && le <= 1) {
        x->learn_rate = le; // Atualiza a taxa de aprendizado
        post("Learning rate: %0.3f", x->learn_rate); // Exibe a nova taxa de aprendizado
    } else {
        error("Learning rate must be between 0 and 1."); // Exibe uma mensagem de erro
    }
}


//------------------------ funções de ativação --------------------------
// Definição das funções de ativação
// Definição das funções de ativação
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x, double target) {
    return x * (1 - x);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x, double target) {
    return x > 0 ? 1 : 0;
}

double prelu(double x) {
    const double alpha = 0.01; // Valor fixo de alpha
    return x > 0 ? x : alpha * x;
}

double prelu_derivative(double x, double target) {
    const double alpha = 0.01; // Valor fixo de alpha
    return x > 0 ? 1 : alpha;
}
double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x, double target) {
    return 1 - tanh(x) * tanh(x);
}

//Função softmax (aplicada a um vetor) [input deve ser vetor_z e output vetor_a da camada que utiliza softmax e length deve ser o tamanho do vetor_z]
double* softmax(double *input, double *output, int length) {
    double max = input[0];
    for (int k = 1; k < length; k++) {
        if (input[k] > max) {
            max = input[k];
        }
    }
    double sum = 0.0;
    for (int j = 0; j < length; j++) {
        output[j] = exp(input[j] - max);
        sum += output[j];
    }
    for (int l = 0; l < length; l++) {
        output[l] /= sum;
        // post("softmax[%d]: %.6f", i, output[i]);
    }
    return output;
}

// Derivada da função softmax (simplificada para uso com cross-entropy)
//argumentos são: saída obtida e saída desejada
double softmax_derivative(double output, double target_output) {
    return output - target_output;
}

double softplus(double x) {
    if (x > 20) { 
        return x; // Para valores grandes, softplus(x) ≈ x
    } else if (x < -20) {
        return exp(x); // Para valores pequenos, evita underflow
    }
    return log(1 + exp(x));
}


double softplus_derivative(double x, double target) {
    return 1 / (1 + exp(-x)); // Derivada da função softplus é a função sigmoid
}



//--------------------------------- Função para calcular entropia cruzada para um exemplo --------------------------------
//argumentos são: p = vetor de classes, q = vetor_a da camada de saída e nº de neurônios na camada de saída
double cross_entropy(double *p, double *q, int size) {
    double loss = 0.0;
    for (int e = 0; e < size; e++) {
        if (p[e] == 1) { // Apenas considerar as classes verdadeiras (p[i] == 1)
            loss -= p[e] * log(q[e]);
        }
    }
    return loss;
}


//------------------------------------- função para selecionar a função de ativação e sua derivada com base no nome -----------------------------------
// Definição dos tipos de função de ativação e suas derivadas
typedef double (*activation_func)(double);
typedef double (*activation_derivative_func)(double, double);

// Funções para obter a função de ativação com base no nome
activation_func get_activation_function(t_symbol *func_name) {
    if (func_name == gensym("sigmoid")) {
        return sigmoid;
    } else if (func_name == gensym("relu")) {
        return relu;
    } else if (func_name == gensym("tanh")) {
        return tanh_activation;
    } else if(func_name == gensym("prelu")) {
        return prelu;
    } else if (func_name == gensym("softmax")) {
        return NULL; // softmax é tratada separadamente
    } else if (func_name == gensym("softplus")){
        return softplus;
    } else {
        return sigmoid; // Padrão
    }
}

// Função para obter a derivada da função de ativação com base no nome
activation_derivative_func get_activation_derivative_function(t_symbol *func_name) {
    if (func_name == gensym("sigmoid")) {
        return sigmoid_derivative;
    } else if (func_name == gensym("relu")) {
        return relu_derivative;
    } else if (func_name == gensym("prelu")) {
        return prelu_derivative;
    } else if (func_name == gensym("tanh")) {
        return tanh_derivative;
    } else if (func_name == gensym("softmax")) {
        return softmax_derivative;
    } else if (func_name == gensym("softplus")) {
        return softplus_derivative;
    } else {
        return sigmoid_derivative; // Padrão
    }
}


//--------------------- Função para configurar as funções de ativação para as camadas específicas ---------------------
void activation_functions(t_mlperceptron *x, t_symbol *s, int argc, t_atom *argv) {
    // Verifica se a quantidade de argumentos é par e se o número de camadas é igual ao número de pares
    if (argc % 2 != 0) {
        error("Please provide pairs of activation function name and layer number.");
        return;
    }

    //verifica se o número de camadas é igual ao número de funções de ativação recebidas
    if (argc / 2 != x->num_hidden) {
        error("Number of activation functions must match the number of hidden layers.");
        return;
    }

    // Atribui a função de ativação a cada camada
    for (int i = 0; i < argc; i += 2) {
        // Verifica se o argumento atual é um símbolo e o próximo é um número
        if (argv[i].a_type != A_SYMBOL || argv[i + 1].a_type != A_FLOAT) {
            error("Activation function is not a symbol or layer number is not a float.");
            return;
        }

        // Obtém o nome da função de ativação e o número da camada
        t_symbol *func_name = atom_getsymbol(&argv[i]);
        int layer = (int)atom_getfloat(&argv[i + 1]);

        // Valida se o número da camada está dentro do intervalo permitido
        if (layer < 1 || layer > x->num_hidden) {  // Ajustado para camadas começando em 1
            error("Layer number %d out of range. Valid range is 1 to %ld.", layer, x->num_hidden);
            continue;
        }

        // Atribui a função de ativação à camada correspondente (ajustando para índice 0)
        x->activation_function[layer - 1] = func_name;
        post("Activation function for layer %d set to: %s", layer, func_name->s_name);
    }
}


// //------------------------- função para liberar memória das matrizes da camada escondida ---------------------
void liberar_hidden_layers(t_mlperceptron *x) {
    if (x->hidden_layers != NULL) {
        for (int m = 0; m < x->num_hidden; m++) {
            if (x->hidden_layers[m] != NULL) {
                // Verifica se o índice m+2 é válido antes de acessar x->net_param
                if (x->net_param != NULL && (m + 2) < x->param) {
                    int linhas = (int)x->net_param[m + 2]; // Obtém o número de linhas da matriz `m`
                    for (int i = 0; i < linhas; i++) {
                        if (x->hidden_layers[m][i] != NULL) {
                            free(x->hidden_layers[m][i]);
                            x->hidden_layers[m][i] = NULL;
                        }
                    }
                }
                free(x->hidden_layers[m]);
                x->hidden_layers[m] = NULL;
            }
        }
        free(x->hidden_layers);
        x->hidden_layers = NULL;
    }
}

//---------------------------- função para liberar memória dos vetores ---------------------
void liberar_vetores(t_mlperceptron *x, double ***array) {
    if (*array != NULL) { //verifica se o array de vetores não é nulo
        for (int i = 0; i < x->num_vetores; i++) { //percorre o array de vetores
            if ((*array)[i] != NULL) { //verifica se cada um dos vetores não é nulo

                free((*array)[i]); // Libera a memória alocada para cada vetor
                (*array)[i] = NULL; // Garante que o ponteiro seja nulo após liberar
            }
        }
        free(*array);  // Libera o array de vetores
        *array = NULL; // Garante que o ponteiro seja nulo após liberar
    }
}

//---------------------------- função para alocar memória dos vetores ----------------------------
int alocar_vetores(t_mlperceptron *x, double ***array, int num_vetores){

    //nº de vetores = nº de camadas (matrizes) da rede (sem considerar a camada de entrada)
    x->num_vetores = num_vetores; 

    *array = (double **)malloc(x->num_vetores * sizeof(double *)); //aloca memória para o array de vetores (número de vetores)

    //verifica se o array de vetores é null
    if (*array == NULL) {
        post("Error allocating memory for vectors");
        return 0;
    }

    // Aloca memória para cada vetor (tamanho do vetor) e verifica se a alocação foi bem-sucedida
    for (int i = 0; i < x->num_vetores; i++) {
        //percorre o array de parâmetros a partir do indice 2 (pula o primeiro e segundo parâmetro que é o nº de camadas e nº de neurônios na camada de entrada)
        int vector_size = (int)x->net_param[i+2];
        (*array)[i] = (double *)malloc(vector_size * sizeof(double)); //aloca memória para cada vetor individualmente  com os nº de neurônios em cada camada
        if ((*array)[i] == NULL) { //verifica individualmente se cada vetor é nulo
            post("Error allocating memory for vector %d.", i);
            liberar_vetores(x, array);  // Libera toda a memória alocada até agora caso dê erro
            return 0;
        }
    }
    return 1;  // Retorna 1 para indicar sucesso
}


//---------------------------- Aloca novas matrizes (camadas escondidas [array de matrizes]) -----------------------------------
int alocar_matrizes(t_mlperceptron *x, int num_matrizes){

    // Configura o número de matrizes
    x->num_hidden = num_matrizes;

    //Aloca memória para o array de matrizes com a quantidade de matrizes fornecida 
    x->hidden_layers = (double ***)malloc(num_matrizes * sizeof(double **));

    //verifica se o array de matrizes é null
    if (x->hidden_layers == NULL) {
        post("Error allocating memory for matrices"); //posta erro caso o array de matrizes for null
        return 0;
    }

    // Aloca individualmente cada matriz com suas dimensões
    for (int m = 0; m < num_matrizes; m++) { //percorre cada matriz
        int colunas = (int)x->net_param[m+1]; //lógica para recuperar os nº de colunas da lista de parâmetros
        int linhas = (int)x->net_param[m+2]; //lógica para recuperar os nº de linhas da lista de parâmetros

         //percorre as matrizes e aloca nº de linhas para cada uma
        x->hidden_layers[m] = (double **)malloc(linhas * sizeof(double *)); //aloca as linhas de cada matriz
        //verifica se as linhas foram alocadas corretamente
        if (x->hidden_layers[m] == NULL) { 
            post("Error allocating memory for rows of the matrix %d.", m); 
            liberar_hidden_layers(x); //libera memória caso ocorra erro na alocação das linhas
            return 0;
        }

        //percorre as linhas de cada matriz e aloca nº de colunas para cada linha
        for(int l = 0; l < linhas; l++){ 
            x->hidden_layers[m][l] = (double *)malloc(colunas * sizeof(double)); // Aloca as colunas de cada linha para cada matriz
        //verifica se as colunas foram alocadas corretamente
            if (x->hidden_layers[m][l] == NULL) { 
                post("Error allocating memory for row %d of the matrix %d.", l, m);
                liberar_hidden_layers(x); //libera memória caso ocorra erro na alocação das colunas
                return 0;
            }
        } 
        post("Weight matrix %d: %d rows and %d columns.", m + 1, linhas, colunas);
    }
    return 1;  // Retorna 1 para indicar sucesso
}


// //-------------------------- preenche os vetores com valores aleatórios ------------------------
void vetor_fill(t_mlperceptron *x, double ***array, t_float lower, t_float higher) {
    if (*array == NULL) {
        error("Os vetores não foram alocados.");
        return;
    }

    srand(time(NULL)); // Inicializa a semente para geração de números aleatórios (uma vez, preferencialmente fora da função)

    for (int v = 0; v < x->num_vetores; v++) { // Percorre o array de vetores
        if ((*array)[v] == NULL) { // Verifica individualmente se cada vetor foi alocado
            post("Error: Vector %d not allocated.\n", v + 1);
            continue; // Pula para o próximo vetor
        }

        int vector_size = (int)x->net_param[v + 2]; // Tamanho do vetor atual
        // post("Bias vector %d filled\n", v + 1);

        for (int k = 0; k < vector_size; k++) {
            (*array)[v][k] = random_double(lower, higher); // Preenche com valores aleatórios de acordo com o intervalo
            // post("%0.2f ", (*array)[v][k]); // Imprime o valor com um espaço para facilitar a leitura
        }
        // post("\n"); // Quebra de linha após cada vetor
    }
    // post("Bias vectors initialized.");
}

//--------------------- preenche as matrizes das camadas escondidas -------------------
void hidden_fill(t_mlperceptron *x, t_float lower, t_float higher) {
    if (x->hidden_layers == NULL) {
        error("Matrices have not been allocated. Use the 'size' method first.");
        return;
    }

    srand(time(NULL)); // Inicia a semente para gerar números aleatórios

    // Preenche cada matriz com valores aleatórios
    for (int m = 0; m < x->num_hidden; m++) {
        int colunas = (int)x->net_param[m+1]; //colunas da matriz m
        int linhas = (int)x->net_param[m+2];// Linhas da matriz `m`

        // post("Weight matrix %d filled:", m + 1);
        for (int i = 0; i < linhas; i++) {
            for (int j = 0; j < colunas; j++) {
                x->hidden_layers[m][i][j] = random_double(lower, higher); // Função que gera número dentro do intervalo
                // post("%0.2f ", x->hidden_layers[m][i][j]); // Exibe o valor gerado
            } 
            // post("\n"); // Nova linha ao final de cada linha da matriz  
            
        } 
        
    }
    // post("Weight matrices initialized.");
}

//-------------------------- inicializa os pesos dos bias com valores aleatórios entre 0 e 1 ------------------------
void random_init(t_mlperceptron *x, t_symbol *s, int argc, t_atom *argv) {

    //verifica se o nº de argumentos é igual a 4 (lower e upper para pesos e bias)
    if(argc != 4) {
        error("Error: please provide a range for random values (lower e upper).");
        return;
    }

    // range dos pesos
    float lower_w = atom_getfloat(&argv[0]); //valor mínimo 
    float higher_w = atom_getfloat(&argv[1]); //valor máximo

    //range do bias
    float lower_b = atom_getfloat(&argv[2]); //valor mínimo
    float higher_b = atom_getfloat(&argv[3]); //valor máximo

    hidden_fill(x, lower_w, higher_w); //preenche as matrizes das camadas escondidas
    vetor_fill(x, &x->vetor_bias, lower_b, higher_b); //preenche os vetores de bias

    post("Weight matrices initialized between %0.2f and %0.2f.", lower_w, higher_w);
    post("Bias vectors initialized between %0.2f and %0.2f.", lower_b, higher_b);
}

//--------------------------- inicialização dos pesos uniforme aleatório ---------------------------
void random_uniforme(t_mlperceptron *x){

    float lower = -1/sqrt(x->net_param[1]);
    float higher = 1/sqrt(x->net_param[1]);

    hidden_fill(x, lower, higher); //preenche as matrizes das camadas escondidas
    vetor_fill(x, &x->vetor_bias, 0, 0.01); //preenche os vetores de bias
    post("Weight initialized between %0.2f and %0.2f.", lower, higher);
    post("Bias vectors initialized between 0 and 0.01.");

}

//--------------------------- inicialização dos pesos método He ---------------------------------------
// ver He et al. (2015) [https://arxiv.org/abs/1502.01852] método utilizado com funções de ativação ReLU
void random_he(t_mlperceptron *x){

    float lower = -sqrt(6/x->net_param[1]);
    float higher = sqrt(6/x->net_param[1]);

    hidden_fill(x, lower, higher); //preenche as matrizes das camadas escondidas
    vetor_fill(x, &x->vetor_bias, 0, 0.01); //preenche os vetores de bias
    post("Weight initialized between %0.2f and %0.2f.", lower, higher);
    post("Bias vectors initialized between 0 and 0.01.");
}

//------------------------------------- inicialização dos pesos método Xavier uniforme --------------------------------------------
// ver Glorot e Bengio (2010) [http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf] método utilizado com funções de ativação sigmoid e tanh
void random_xavier(t_mlperceptron *x) {
    for (int m = 0; m < x->num_hidden; m++) {
        int input_neurons = (int)x->net_param[m + 1]; //nº de neurônios na camada de entrada
        int output_neurons = (int)x->net_param[m + 2]; //nº de neurônios na camada de saída

        double lower = -sqrt(6.0 / (input_neurons + output_neurons));
        double higher = sqrt(6.0 / (input_neurons + output_neurons));

        for (int i = 0; i < output_neurons; i++) {
            for (int j = 0; j < input_neurons; j++) {
                hidden_fill(x, lower, higher); //preenche as matrizes das camadas escondidas
            }
        }
        post("Weight matrix %d initialized between %0.2f and %0.2f.", m + 1, lower, higher);
    }

    vetor_fill(x, &x->vetor_bias, 0, 0.01); //preenche os vetores de bias
    post("Bias vectors initialized between 0 and 0.01.");
}

//---------------------------------- inicialização dos pesos método Lecun ------------------------------
//ver 
void random_lecun(t_mlperceptron *x){
    for (int m = 0; m < x->num_hidden; m++) {
        int input_neurons = (int)x->net_param[m + 1]; //nº de neurônios na camada de entrada
        int output_neurons = (int)x->net_param[m + 2]; //nº de neurônios na camada de saída

        double lower = -sqrt(1.0 / (input_neurons));
        double higher = sqrt(1.0 / (input_neurons));

        for (int i = 0; i < output_neurons; i++) {
            for (int j = 0; j < input_neurons; j++) {
                hidden_fill(x, lower, higher); //preenche as matrizes das camadas escondidas
            }
        }
        post("Weight matrix %d initialized between %0.2f and %0.2f.", m + 1, lower, higher);
    }

    vetor_fill(x, &x->vetor_bias, 0, 0.01); //preenche os vetores de bias
    post("Bias vectors initialized between 0 and 0.01.");
}

//----------------------------------- Redimensiona as matrizes com tamanhos distintos -----------------------------
void matriz_size(t_mlperceptron *x, t_symbol *s, int argc, t_atom *argv) {

    // Verifica se há argumentos suficientes
    if (argc <= 0 || argv == NULL) {
        error("Error: No arguments provided.");
        return;
    }

    // Libera memória das matrizes e vetores e define como null antes de mudar os parâmetros da rede
    //[IMPORTANTE] Isso é necessário para evitar vazamento de memória
    liberar_vetores(x, &x->vetor_bias); //vetor de bias 
    liberar_vetores(x, &x->vetor_a); //vetor de ativação
    liberar_vetores(x, &x->vetor_z); //vetor de soma ponderada
    liberar_vetores(x, &x->delta); //vetor de gradientes
    liberar_hidden_layers(x); //matrizes com pesos de cada camadas da rede

    //libera memória do array de parâmetros e define como null
    if (x->net_param != NULL) { //verifica se o array de parâmetros não é nulo
        free(x->net_param);  // Libera a memória alocada se o array não for null
        x->net_param = NULL; // define o array como null para evitar ponteiros que apontam para um local de memória inválido ou desalocado
    }

    //define o número de argumentos recebidos
    x->param = (int)argc; 


    // aloca memória para o array de parâmetros com o tamanho da lista de argumentos recebida
    x->net_param = (t_float *)malloc(x->param * sizeof(t_float)); 


    //verifica se a alocação foi bem-sucedida
    if(x->net_param == NULL) { 
        error("Error allocating memory for parameters vector.");
        return;
    }

    // libera memória do array de funções de ativação e define como null
    if(x->activation_function != NULL){
        freebytes(x->activation_function, x->num_hidden * sizeof(t_symbol *));
        x->activation_function = NULL;
    }

    // copia os argumentos para o array net_param
    for (int i = 0; i < x->param; i++) {
        if (argv[i].a_type != A_FLOAT) {
            error("Error: All arguments must be of type float.");
            free(x->net_param);
            x->net_param = NULL;
            return;
        }
        x->net_param[i] = (int)argv[i].a_w.w_float; //copiar os argumentos para o array net_param
    }

    //verifica se o vetor de entrada é nulo, se não for, libera a memória alocada anteriormente
    if(x->input_data != NULL){
        free(x->input_data); // Libera a memória alocada anteriormente
        x->input_data = NULL; //define o vetor de entrada como null
        //aloca memória para o vetor de entrada com tamanho do nº de dimensões dos dados de entrada
        x->input_data = (double *)malloc(x->net_param[1] * sizeof(double)); 
        post("Input data dimension: %d", (int)x->net_param[1]);
    }


    //verifica se o array de classes é nulo, se não for, libera a memória alocada anteriormente
    if(x->classes != NULL){
        free(x->classes); // Libera a memória alocada anteriormente
        x->classes = NULL; //define o vetor de classes como null
        //aloca memória para o vetor de classes com tamanho do nº de neurônios da camada de saída
        x->classes = (double *)malloc(x->net_param[x->param-1] * sizeof(double)); 
        post("Classes: %d", (int)x->net_param[x->param-1]);
    }

    //verifica se o array de erro por classes é nulo, se não for, libera a memória alocada anteriormente
    if(x->class_erro != NULL){
        free(x->class_erro); // Libera a memória alocada anteriormente
        x->class_erro = NULL; //define o array de erro por classes como null
        //aloca memória para o array de erros por classes com tamanho do nº de neurônios da camada de saída
        x->class_erro = (t_float *)malloc(x->net_param[x->param-1] * sizeof(t_float)); 
        for(int c = 0; c < x->net_param[x->param-1]; c++){ //inicia todos valores como 0
            x->class_erro[c] = 0;
        }
    }


   // verifica se o nº de argumentos recebidos é menor que 3 (redes devem ter ao menos 3 parâmetros [camadas, dimensão dos dados e neurônios]) 
   //e se os parâmetros correspondem ao número de camadas fornecido
    if (x->param < 3 || (x->param - 2 != (int)x->net_param[0])) {
        error("Error: 'size' requires the number of layers, input data dimension, and amount of neurons in each layer.");
        free(x->net_param);
        x->net_param = NULL;
        return;
    }
    
    // Verifica se os parâmetros recebidos são maiores que zero
    for (int k = 0; k < x->param; k++) {
        if ((int)x->net_param[k] <= 0) {
            error("Error: size parameters must be greater than zero");
            free(x->net_param);
            x->net_param = NULL;
            return;
        }
    }
    
    //atribui o primeiro parâmetro (nº de camadas) como nº de matrizes 
    int num_matrizes = (int)x->net_param[0]; // Converte o primeiro parâmetro para inteiro 

    // garante que nº de matrizes (camadas) é maior que zero
    if (num_matrizes <= 0) {
        error("Error: the number of layers must be greater than zero");
        free(x->net_param);
        x->net_param = NULL;
        return;
    }

    // verifica se o nº de dimensões corresponde ao nº de matriz
    int num_dimensoes = (int)x->param - 1;
    if (num_dimensoes != num_matrizes + 1) { //verifica se o nº de dimensões corresponde ao nº de matrizes
        error("Error: the number of arguments provided does not match number of layers.");
        free(x->net_param);
        x->net_param = NULL;
        return;
    }
    
    //------------------- realocação das matrizes com novas dimensões ----------------------
    if (!alocar_matrizes(x, num_matrizes)) {
        error("Error resizing weight matrices. Allocation failed.");
        return;
    }
    
    //----------------------- realocação dos vetores de bias com novos tamanhos ---------------------
    if(!alocar_vetores(x, &x->vetor_bias, num_matrizes)){
        error("Error resizing bias vectors. Allocation failed.");
        return;
    }

    
    //----------------------- realocação dos vetores da soma ponderada com novos tamanhos ---------------------
    if(!alocar_vetores(x, &x->vetor_z, num_matrizes)){
        error("Error resizing vectors z. Allocation failed.");
        return;
    }
    // for(int v = 0; v < num_matrizes; v++){
    //     int vector_size = (int)x->net_param[v+2]; //tamanho do vetor
        // post("Z vector %d size: %d.", v + 1, vector_size);
    // }

    //----------------------- realocação dos vetores de ativação com novos tamanhos ---------------------
    if(!alocar_vetores(x, &x->vetor_a, num_matrizes)){
        error("Error resizing activation vectors. Allocation failed.");
        return;
    }
    // for(int v = 0; v < num_matrizes; v++){
    //     int vector_size = (int)x->net_param[v+2]; //tamanho do vetor
    //     post("Activation vector %d size: %d.", v + 1, vector_size);
    // }


    //----------------------- realocação dos vetores de gradientes com novos tamanhos ---------------------
    if(!alocar_vetores(x, &x->delta, num_matrizes)){
        error("Error resizing delta vectors. Allocation failed.");
        return;
    }

    //preenche os vetores de gradientes com zeros
    for (int m = 0; m < x->num_hidden; m++) { //percorre as camadas
        int linhas = (int)x->net_param[m + 2]; //nº de neurônios na camada (nº de linhas da matriz)
        for (int i = 0; i < linhas; i++) {
            x->delta[m][i] = 0; //preenche o vetor de gradientes com zeros
        }
    }

    // --------------- preenche as matrizes com novos valores aleatórios -------------------
    if (x->hidden_layers != NULL) {
        hidden_fill(x,0, 1);
    } else {
        error("Error: Hidden layers not allocated.");
    }

    //---------------------- preeche os vetores de bias com novos valores aleatórios -------------------
    if (x->vetor_bias != NULL) {
        vetor_fill(x, &x->vetor_bias, 0, 0.01);
    } else {
        error("Error: Bias vectors not allocated.");
    }


    //------------------ aloca memória para o array de funções de ativação ---------------------------
    x->activation_function = (t_symbol **)getbytes(x->num_hidden * sizeof(t_symbol *));
    for (int i = 0; i < x->num_hidden; i++) {
        x->activation_function[i] = gensym("sigmoid"); // Função de ativação padrão para todas as camadas
    }


    //------------------ aloca memória para o buffer de saída -------------------
    // Aloca memória para o buffer x->x_layer_out com o tamanho do nº de neurônios da camada de saída
    if (x->x_layer_out != NULL) {//verifica se o buffer de saída não é nulo
        freebytes(x->x_layer_out, x->net_param[x->param-1] * sizeof(t_atom));//libera a memória alocada anteriormente
        x->x_layer_out = NULL;//define o buffer de saída como null
        }
    // Aloca memória para o buffer de saída que será enviado para o outlet (saída da última camada)
    x->x_layer_out = (t_atom *)getbytes(x->net_param[x->param-1] * sizeof(t_atom)); 
    // post("Output buffer size: %d", (int)x->net_param[x->param-1]);

}


//------------------------- retorna os valores das matrizes das camadas escondidas no outlet 1 -------------------
void matrizes_out(t_mlperceptron *x) {
    if (x->hidden_layers != NULL) {
        int offset = 0; // Índice para percorrer a lista de parâmetros (net_param)
        for (int m = 0; m < x->num_hidden; m++) { //percorre as matrizes
            int colunas = x->net_param[offset + 1];     // Número de colunas da matriz
            int linhas = x->net_param[offset + 2]; // Número de linhas da matriz
            offset += 1; // Atualiza o deslocamento para a próxima matriz

            post("Weigth matrix %d", m + 1);

            // Aloca memória para x->x_linha_out, se necessário
            if (x->x_linha_out != NULL) {
                freebytes(x->x_linha_out, colunas * sizeof(t_atom));
                x->x_linha_out = NULL;
            }
                // Aloca memória para a linha atual
            x->x_linha_out = (t_atom *)getbytes(colunas * sizeof(t_atom)); 

            for (int k = 0; k < linhas; k++) { // Percorre as linhas da matriz
                // Preenche o buffer de saída com os valores da linha atual
                for (int j = 0; j < colunas; j++) {
                    SETFLOAT(x->x_linha_out + j, x->hidden_layers[m][k][j]);
                }
                // Envia a linha atual da matriz para o outlet
                // outlet_list(x->x_out2, &s_list, colunas, x->x_linha_out);
                outlet_anything(x->x_out2, gensym("weight"), colunas, x->x_linha_out);
            }
        }
        // Libera memória do buffer de saída após o uso
        if (x->x_linha_out != NULL) {
            freebytes(x->x_linha_out, sizeof(t_atom));
            x->x_linha_out = NULL;
        }
    } else {
        post("Error: Hidden layers not allocated.");
    }
}

//------------------------- retorna os valores dos vetores de bias no outlet 2 -------------------
void vetores_out(t_mlperceptron *x) {
    if (x->vetor_bias != NULL) {
        for (int v = 0; v < x->num_vetores; v++) { // Percorre os vetores
            int vector_size = (int)x->net_param[v + 2]; // Tamanho do vetor

            post("Bias vector %d", v + 1);

            // Aloca memória para x->x_bias_out, se necessário
            if (x->x_bias_out != NULL) {
                freebytes(x->x_bias_out, vector_size * sizeof(t_atom));
                x->x_bias_out = NULL;
            }
            // Aloca memória para o buffer de saída
            x->x_bias_out = (t_atom *)getbytes(vector_size * sizeof(t_atom)); 

            // Preenche o buffer de saída com os valores do vetor de bias
            for (int j = 0; j < vector_size; j++) {
                SETFLOAT(x->x_bias_out + j, x->vetor_bias[v][j]);
            }
            // Envia o buffer de saída para o outlet
            outlet_anything(x->x_out2, gensym("bias"), vector_size, x->x_bias_out);
        }
        // Libera memória do buffer de saída após o uso
        if (x->x_bias_out != NULL) {
            freebytes(x->x_bias_out, sizeof(t_atom));
            x->x_bias_out = NULL;
        }
    } else {
        post("Error: Bias vectors not allocated.");
    }
}

//------------------------- retropropagação do erro ----------------------------------
void backpropagation(t_mlperceptron *x, double *target_output) {
    // Calcular o erro na camada de saída
    int output_layer = x->num_hidden - 1;
    int num_output_neurons = (int)x->net_param[x->param - 1];
    t_symbol *output_activation = x->activation_function[output_layer]; //

    // Obter a função de derivada de ativação para a camada de saída
    activation_derivative_func activation_derivative = get_activation_derivative_function(output_activation);

    //calcula o delta da camada de saída
    for (int i = 0; i < num_output_neurons; i++) { //percorre os neurônios da camada de saída
        double output = x->vetor_a[output_layer][i];// pega os valores do vetor de ativação da camada de saída
        double error; //variável para armazenar o erro
        if (output_activation == gensym("softmax")) { //verifica se a função de ativação da camada de saída é softmax
            // Usar entropia cruzada como função de custo
            //calcula o delta da camada de saída para função softmax
            x->delta[output_layer][i] = target_output[i] - output;
            // post("delta camada saída[%d]: %.6f", i, x->delta[output_layer][i]);
        } else {
            // Usar erro quadrático médio como função de custo
            error = target_output[i] - output;
            x->delta[output_layer][i] = error * activation_derivative(output, 0); // Usar a função de derivada apropriada
        }
        x->error += error * error;
        // post("Delta out[%d][%d]: %0.6f", output_layer, i, x->delta[output_layer][i]);   
    }

    if (output_activation == gensym("softmax")){
        x->cross_entropy += cross_entropy(x->classes, x->vetor_a[output_layer], (int)x->net_param[x->param - 1]); //calcula entropia cruzada
        }

    // Propagar o erro para trás através das camadas ocultas
    for (int m = x->num_hidden - 2; m >= 0; m--) {
        int linhas = (int)x->net_param[m + 2];
        int colunas = (int)x->net_param[m + 1];
        activation_derivative_func activation_derivative = get_activation_derivative_function(x->activation_function[m]);

        for (int i = 0; i < linhas; i++) {
            double sum = 0;
            for (int j = 0; j < (int)x->net_param[m + 3]; j++) {
                sum += x->delta[m + 1][j] * x->hidden_layers[m + 1][j][i];
            }
            x->delta[m][i] = sum * activation_derivative(x->vetor_a[m][i], 0); // Passar dois argumentos
            // post("Delta[%d][%d]: %0.6f", m, i, x->delta[m][i]);
        }
    }

    // Atualizar os pesos e os bias usando os gradientes calculados e a taxa de aprendizado
    for (int m = 0; m < x->num_hidden; m++) {
        int linhas = (int)x->net_param[m + 2];
        int colunas = (int)x->net_param[m + 1];
        double *input_current = (m == 0) ? x->input_data : x->vetor_a[m - 1];

        for (int i = 0; i < linhas; i++) {
            for (int j = 0; j < colunas; j++) {
                x->hidden_layers[m][i][j] += x->learn_rate * x->delta[m][i] * input_current[j];
                // post("Weight[%d][%d][%d]: %0.6f", m, i, j, x->hidden_layers[m][i][j]);
            }
            x->vetor_bias[m][i] += x->learn_rate * x->delta[m][i];
            // post("Bias[%d][%d]: %0.6f", m, i, x->vetor_bias[m][i]);
        }
    }
}


//------------------------- propagação direta ---------------------------------------------
void forward_propagation(t_mlperceptron *x) {

    // Propagação dos dados
    for (int m = 0; m < x->num_hidden; m++) {
        int colunas = (int)x->net_param[m + 1];
        int linhas = (int)x->net_param[m + 2];

        // verifica se a camada atual [m] é a primeira (m=0), se sim, input_current = dados de entrada
        //caso contrário (m!=0), input_current = vetor de ativação da camada anterios (x->vetor_a[m-1])
        double *input_current = (m == 0) ? x->input_data : x->vetor_a[m - 1];

        activation_func activation = get_activation_function(x->activation_function[m]); //percorre o array de funções de ativação e as define para cada camada
        if (activation == NULL) { 
            //verifica se activation é NULL, se sim, verifica se a camada atual é a camada de saída
            //e se a função de ativação da camada atual é softmax, se for verdade aplica softmax
            if(m == x->num_hidden -1 && x->activation_function[m] == gensym("softmax")){
                // Softmax precisa de vetor_z calculado
                for (int i = 0; i < linhas; i++) {
                    x->vetor_z[m][i] = 0;
                    for (int j = 0; j < colunas; j++) {
                        x->vetor_z[m][i] += x->hidden_layers[m][i][j] * input_current[j]; //soma ponderada
                    }
                    x->vetor_z[m][i] += x->vetor_bias[m][i];
                    // post("vetor z camada saída: %0.6f", x->vetor_z[m][i]); 
                    // Aplicar softmax diretamente no vetor_z e gravar no vetor_a da camada de saída
                    softmax(x->vetor_z[m], x->vetor_a[m], (int)x->net_param[x->param - 1]);
                    // post("vetor de ativação camada saída: %0.6f", x->vetor_a[m][i]);        
                }
            } else {
                error("Activation function for layer %d is NULL.", m);
                return;
            }
        } else {
            for (int i = 0; i < linhas; i++) { // Percorre cada neurônio da camada
                x->vetor_z[m][i] = 0; // Inicializa soma ponderada
                for (int j = 0; j < colunas; j++) {
                    x->vetor_z[m][i] += x->hidden_layers[m][i][j] * input_current[j]; // Soma ponderada
                }
                x->vetor_z[m][i] += x->vetor_bias[m][i]; // Adiciona bias
                x->vetor_a[m][i] = activation(x->vetor_z[m][i]);
                // post("vetor de ativação camada [%d], neurônio [%d]: %0.6f", m, i, x->vetor_a[m][i]);
            }
        }
    }

        // Copia o vetor_a da camada de saída para o buffer layer_out
        int output_layer = x->num_hidden - 1;
        int num_output_neurons = (int)x->net_param[x->param - 1];

        for (int i = 0; i < num_output_neurons; i++) {
            SETFLOAT(x->x_layer_out + i, x->vetor_a[output_layer][i]);
            // post("vetor a saída, neurônio [%d]: %0.6f", i, x->vetor_a[output_layer][i]);
        }
        // Envia o buffer de saída para o outlet
        outlet_list(x->x_out1, &s_list, num_output_neurons, x->x_layer_out);
}


//---------------------------------------- avalia o modelo -----------------------------------------------
void evaluation(t_mlperceptron *x){
    if (x->current_data <= x->datasize){
        forward_propagation(x);
        //realiza a avaliação do modelo treinado
        int output_layer = x->num_hidden - 1;
        int num_output_neurons = (int)x->net_param[x->param - 1];
        int max_index = 0;
        double max_value = x->vetor_a[output_layer][0];
                
        //percorre o vetor de saída e procura o maior valor
        for(int a = 0; a < num_output_neurons; a++){
            if (x->vetor_a[output_layer][a] > max_value) {
                max_value = x->vetor_a[output_layer][a];
                max_index = a; //índice do maior valor
            }
            x->vetor_a[output_layer][a] = 0; //menores valores atribuídos como 0
        }
        x->vetor_a[output_layer][max_index] = 1; //maior valor atribuído como 1
        //percorre e compara o vetor de saída com as classes corretas
        for(int k = 0; k < num_output_neurons; k++){
            if (x->vetor_a[output_layer][k] == 1 && x->vetor_a[output_layer][k] != x->classes[k]){
                x->class_erro[k] += 1; //soma o erro para cada classe
                x->num_erro += 1; //soma o erro total
            }
        }
    }
    x->current_data ++; //incrementa o contador de exemplos
    post("Test example %d", x->current_data);
    if (x->current_data == x->datasize){ //retorna as informações de erro quando a avaliação terminar
        int num_output_neurons = (int)x->net_param[x->param - 1];
        for(int e = 0; e < num_output_neurons; e++){
            post("Errors classifying class %d: %d", e, (int)x->class_erro[e]);
        }
        float erro_percent = x->num_erro * 100 / x->datasize;
        float acuracy = 100 - erro_percent;
        // Exibe a taxa de erro total
        post("Total classification errors: %d", (int)x->num_erro);
        post("Percentage of error: %0.2f", erro_percent);
        post("Acuracy: %0.2f", acuracy);
        for(int j = 0; j < num_output_neurons; j++){
            x->class_erro[j] = 0;
        }
        x->num_erro = 0;
        x->current_data = 0;
    }
   
}



//----------------------------- treinamento da rede ------------------------------------
void training_data(t_mlperceptron *x, t_symbol *s, int argc, t_atom *argv) {
    static int message_printed = 0; // Variável de controle para rastrear se a mensagem já foi impressa

    //--------------- recebe os dados de entrada e as classes ----------------------
    // Verifica se o tamanho dos dados recebidos é igual à dimensão dos dados de entrada + nº de neurônios de saída
    if (argc != x->net_param[1] + x->net_param[x->param - 1]) {
        error("Incompatible input data dimension.");
        return;
    }
    // Preenche o vetor de entrada com os valores fornecidos
    if (x->input_data != NULL) {
        for (int i = 0; i < argc - x->net_param[x->param - 1]; i++) {
            x->input_data[i] = argv[i].a_w.w_float;
            // post("input data %d: %0.2f", i, x->input_data[i]);
        }
    }

    // Preenche o vetor de classes com os valores fornecidos
    if (x->classes != NULL) {
        int num_classes = (int)x->net_param[x->param - 1];
        int start_index = argc - num_classes;

        for (int c = 0; c < num_classes; c++) {
            x->classes[c] = argv[start_index + c].a_w.w_float;
            // post("classes %d: %d", c, (int)x->classes[c]);
        }
    }

    //------------------------ propaga os dados pelas camadas, retropropaga e atualiza os pesos --------------------------------  
    if (x->current_epoch <= x->max_epochs && x->current_data < x->datasize && x->trainingmode == 1 && x->evalmode == 0) {
        forward_propagation(x);
        x->current_data++;
        backpropagation(x, x->classes);
    } else if (x->trainingmode == 0 && x->evalmode == 0){
        forward_propagation(x);
    }
    //se o índice do exemplo atual é igual ao número máximo de exemplos adicona uma época no contador
    if (x->current_data == x->datasize && x->trainingmode == 1) {
        x->current_data = 0;
        x->current_epoch++;
        if(x->activation_function[x->num_hidden - 1] == gensym("softmax")){ //verifica se é softmax na camada de saída
            double avg_entropy = x->cross_entropy / x->datasize; //calcula a média da entropia cruzada
            SETFLOAT(x->x_mse, avg_entropy);
            outlet_anything(x->x_out2, gensym("error"), 1, x->x_mse);
            post("Epoch %d - Average Cross-Entropy: %.6f", x->current_epoch, avg_entropy);
            x->cross_entropy = 0;
        }
        else{ //para outras funções de ativação na camada de saída, calcula o mse
            double avg_mse = x->error / x->datasize;
            SETFLOAT(x->x_mse, x->error);
            outlet_anything(x->x_out2, gensym("error"), 1, x->x_mse);
            post("Epoch: %d - MSE: %0.3f", x->current_epoch, x->error);
            x->error = 0;
        }
     }
        //verifica se o treinamento atingiu o número máximo de épocas
    if (x->current_epoch == x->max_epochs) {
        if (!message_printed) { // Verifica se a mensagem já foi impressa
            post("The training process has reached the maximum amount of epochs (%d/%d)", x->current_epoch, x->max_epochs);
            message_printed = 1; // Define a variável de controle para indicar que a mensagem foi impressa
            x->trainingmode = 0;
            x->current_epoch = 0;
            x->current_data = 0;   
        }          
    }

    //avalia o modelo treinado se o modo de avaliação está on (= 1)
    if (x->trainingmode == 0 && x->evalmode == 1){
         evaluation(x);
    }
}

//-------------------------------------- reseta os parâmetros da rede  --------------------------------------
static void reset(t_mlperceptron *x){
    //redefine hiperparâmetros e reinicializa os vetores, matrizes e funções de ativação sigmoid para todas camadas
    x->trainingmode = 0;
    x->evalmode = 0;
    x->learn_rate = 0.1;
    x->max_epochs = 100;
    x->current_epoch = 0;
    x->current_data = 0;
    x->datasize = 10;
    x->error = 0;
    x->cross_entropy = 0;
    

    hidden_fill(x, 0, 1);
    vetor_fill(x, &x->vetor_bias, 0, 0.01);
    vetor_fill(x, &x->vetor_a, 0, 0);
    vetor_fill(x, &x->vetor_z, 0, 0);
    vetor_fill(x, &x->delta, 0, 0);

    // Inicializa o array de funções de ativação
    if(x->activation_function != NULL){
        freebytes(x->activation_function, x->num_hidden * sizeof(t_symbol *));
        x->activation_function = NULL;
    }
    x->activation_function = (t_symbol **)getbytes(x->num_hidden * sizeof(t_symbol *));
    for (int i = 0; i < x->num_hidden; i++) {
        x->activation_function[i] = gensym("sigmoid");  // Função de ativação padrão para todas as camadas
        // post("Activation function %d: %s", i, x->activation_function[i]->s_name);
    }
    post("Neural network reseted.");

} 

//-------------------------------- salva o modelo treinado -------------------------------
static void model_save(t_mlperceptron *x, t_symbol *filename)
{
    FILE *fd;
    char buf[MAXPDSTRING];
    canvas_makefilename(x->x_canvas, filename->s_name, buf, MAXPDSTRING); // Cria o nome do arquivo
    sys_bashfilename(buf, buf); // Converte o nome do arquivo para o formato do sistema
    if (!(fd = fopen(buf, "w"))) // Tenta abrir o arquivo para escrita
    {
        error("%s: can't create", buf);
        return;
    }

    // Salvar a estrutura da rede
    fprintf(fd, "Network Structure:\n");
    fprintf(fd, "Parameters: %ld\n", x->param);
    fprintf(fd, "Number of layers: %ld\n", x->num_hidden);
    for (int i = 0; i < x->param; i++) {
        fprintf(fd, "%g ", x->net_param[i]);
    }
    fprintf(fd, "\n");

    // Salvar as funções de ativação
    fprintf(fd, "Activation Functions:\n");
    for (int i = 0; i < x->num_hidden; i++) {
        fprintf(fd, "%s ", x->activation_function[i]->s_name);
    }
    fprintf(fd, "\n");

    // Salvar os hiperparâmetros
    fprintf(fd, "Hyperparameters:\n");
    fprintf(fd, "Learning rate: %g\n", x->learn_rate);
    fprintf(fd, "Max epochs: %ld\n", x->max_epochs);
    fprintf(fd, "Current epoch: %ld\n", x->current_epoch);
    fprintf(fd, "Training mode: %g\n", x->trainingmode);
    fprintf(fd, "Data size: %ld\n", x->datasize);
    fprintf(fd, "Error: %g\n", x->error);

    // Salvar os pesos das camadas ocultas
    for (int m = 0; m < x->num_hidden; m++) {
        int linhas = (int)x->net_param[m + 2];
        int colunas = (int)x->net_param[m + 1];
        fprintf(fd, "Layer %d Weights:\n", m + 1);
        for (int i = 0; i < linhas; i++) {
            for (int j = 0; j < colunas; j++) {
                fprintf(fd, "%g ", x->hidden_layers[m][i][j]);
            }
            fprintf(fd, "\n");
        }
    }

    // Salvar os bias das camadas ocultas
    for (int m = 0; m < x->num_hidden; m++) {
        int linhas = (int)x->net_param[m + 2];
        fprintf(fd, "Layer %d Bias:\n", m + 1);
        for (int i = 0; i < linhas; i++) {
            fprintf(fd, "%g ", x->vetor_bias[m][i]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
    x->trainingmode = 0;
    post("Model saved to %s", buf);
}


//-------------------------------- carrega o modelo treinado -------------------------------
static void model_load(t_mlperceptron *x, t_symbol *filename, t_symbol *format) {
    FILE *fd;
    char buf[MAXPDSTRING];
    char line[1024];
    canvas_makefilename(x->x_canvas, filename->s_name, buf, MAXPDSTRING);
    sys_bashfilename(buf, buf);

    if (!(fd = fopen(buf, "r"))) {
        error("%s: can't open", buf);
        return;
    }

    // Liberação de memória previamente alocada
    liberar_vetores(x, &x->vetor_bias);
    liberar_vetores(x, &x->vetor_a);
    liberar_vetores(x, &x->vetor_z);
    liberar_vetores(x, &x->delta);
    liberar_hidden_layers(x);

    if (x->activation_function != NULL) {
        freebytes(x->activation_function, x->num_hidden * sizeof(t_symbol *));
        x->activation_function = NULL;
    }

    if (x->net_param != NULL) {
        free(x->net_param);
        x->net_param = NULL;
    }

    if (x->input_data != NULL) {
        free(x->input_data);
        x->input_data = NULL;
    }

    if (x->classes != NULL) {
        free(x->classes);
        x->classes = NULL;
    }

    if (x->class_erro != NULL) {
        free(x->class_erro);
        x->class_erro = NULL;
    }


    // Ler a primeira linha do arquivo
    if (fgets(line, sizeof(line), fd) != NULL) {
        // Imprimir a linha lida
        post("%s", line);
    } else {
        fprintf(stderr, "Error reading first line of file\n");
    }

    // Ler e processar cada linha
    if (fgets(line, sizeof(line), fd) && sscanf(line, "Parameters: %ld", &x->param) == 1) {
        post("Parameters: %ld\n", x->param);
    }

    if (fgets(line, sizeof(line), fd) && sscanf(line, "Number of layers: %ld", &x->num_hidden) == 1) {
        post("Number of layers: %ld\n", x->num_hidden);
    }

    // Aloca memória para `x->net_param`
    x->net_param = (t_float *)malloc(x->param * sizeof(t_float));
    if (!x->net_param) {
        post("Error allocating memory for parameters vector");
        return;
    }

    if (fgets(line, sizeof(line), fd)) {
        int idx = 0; // Índice para x->net_param
        char *ptr = line; // Ponteiro para processar a linha
        int offset; // Variável para armazenar o número de caracteres lidos

        while (sscanf(ptr, "%f%n", &x->net_param[idx], &offset) == 1) {
            post("size: %f\n", x->net_param[idx]);
            ptr += offset; // Avança o ponteiro pela quantidade lida
            idx++; // Incrementa o índice do vetor
        }
    }

    //atribui o número de camadas
    x->num_hidden = (int)x->net_param[0]; 

    //aloca memória para o array de funções de ativação
    x->activation_function = (t_symbol **)getbytes(x->num_hidden * sizeof(t_symbol *));

    // Lê a linha de "Activation Functions:" e a ignora
    if (fgets(line, sizeof(line), fd)) {
        // post("Cabeçalho de funções de ativação: %s", line); // Apenas para debug
    }

    // Lê a linha com as funções de ativação
    if (fgets(line, sizeof(line), fd)) {
        // post("Funções de ativação: %s", line); // Apenas para debug

        // Tokeniza a linha para pegar cada função
        char *token = strtok(line, " \n");
        int i = 0;

        // Lê os tokens da linha e atribui ao array
        while (token && i < x->num_hidden) {
            x->activation_function[i] = gensym(token); // Converte para t_symbol
            // post("Função de ativação %d: %s", i, token); // Debug
            token = strtok(NULL, " \n");
            i++;
        }

        // Verifica se o número de funções lidas é o esperado
        if (i != x->num_hidden) {
            post("Number of activation functions read (%d) does not match expectations", i, x->num_hidden);
        }
    }
    //imprime as funções de ativação lidas
    for(int f = 0; f < x->num_hidden; f++){
        post("Activation function layer %d: %s", f+1, x->activation_function[f]->s_name);
    }

    // Lê a linha de "hiperparameter:" e a ignora
    if (fgets(line, sizeof(line), fd)) {
        post("%s", line); // Apenas para debug
    }
    //lê a taxa de aprendizagem
    if (fgets(line, sizeof(line), fd) && sscanf(line, "Learning rate: %f", &x->learn_rate) == 1) {
        post("Learning rate: %f\n", x->learn_rate);
    }

    //lê o nº máximo de épocas
    if (fgets(line, sizeof(line), fd) && sscanf(line, "Max epochs: %ld", &x->max_epochs) == 1) {
        post("Max epochs: %ld\n", x->max_epochs);
    }

    //lê a época atual
    if (fgets(line, sizeof(line), fd) && sscanf(line, "Current epoch: %ld", &x->current_epoch) == 1) {
        post("Current epoch: %ld\n", x->current_epoch);
    }

    //lê o modo de treinamento
    if (fgets(line, sizeof(line), fd) && sscanf(line, "Training mode: %f", &x->trainingmode) == 1) {
        post("Training mode: %d\n", x->trainingmode);
    }

    //lê o nº de exemplos de treinamento
    if (fgets(line, sizeof(line), fd) && sscanf(line, "Data size: %ld", &x->datasize) == 1) {
        post("Amount of training example: %ld\n", x->datasize);
    }

    //lê o erro dá última época
    if (fgets(line, sizeof(line), fd) && sscanf(line, "Error: %f", &x->error) == 1) {
        post("Error in the last training epoch: %f\n", x->error);
    }


    // Alocar memória para vetores auxiliares
    x->input_data = (double *)malloc(x->net_param[1] * sizeof(double));
    x->classes = (double *)malloc(x->net_param[x->param - 1] * sizeof(double));
    x->class_erro = (t_float *)malloc(x->net_param[x->param - 1] * sizeof(t_float));

    
    // Alocar estruturas dinâmicas
    if (!alocar_matrizes(x, x->num_hidden) ||
        !alocar_vetores(x, &x->vetor_bias, x->num_hidden) ||
        !alocar_vetores(x, &x->vetor_a, x->num_hidden) ||
        !alocar_vetores(x, &x->vetor_z, x->num_hidden) ||
        !alocar_vetores(x, &x->delta, x->num_hidden)) {
        error("Error allocating memory for network structures (matrices and vectors)");
        fclose(fd);
        return;
    }

    // Lê os pesos das camadas ocultas
    int m = -1; // Índice da camada atual
    int l = 0;  // Índice da linha atual para pesos

    while (fgets(line, sizeof(line), fd)) {
        // Verifica se é uma linha de cabeçalho de pesos
        if (strstr(line, "Layer") && strstr(line, "Weights")) {
            sscanf(line, "Layer %d Weights:", &m); // Lê o índice da camada
            m--; // Ajusta o índice para o array
            l = 0; // Reinicia o índice de linhas para os pesos
            post("Reading weights of layer %d", m + 1);
            continue;
        }

        // Processa os pesos
        if (m >= 0 && l < (int)x->net_param[m + 2]) {
            char *ptr = line; // Ponteiro para a linha atual
            for (int c = 0; c < (int)x->net_param[m + 1]; c++) {
                if (sscanf(ptr, "%lf", &x->hidden_layers[m][l][c]) == 1) {
                    ptr = strchr(ptr, ' '); // Avança para o próximo número
                    if (!ptr) break;       // Para se não houver mais números
                    ptr++;                 // Move para o próximo caractere após o espaço
                }
            }
            l++; // Avança para a próxima linha de pesos
        }
    }

    // // Reinicia o arquivo para ler os bias
    rewind(fd);

     // Reinicia o índice para ler os bias
    m = -1;
    l = 0;
    // Lê os bias das camadas ocultas
    while (fgets(line, sizeof(line), fd)) {
        // Verifica se é uma linha de cabeçalho de bias
        if (strstr(line, "Layer") && strstr(line, "Bias")) {
            sscanf(line, "Layer %d Bias:", &m); // Lê o índice da camada
            m--; // Ajusta o índice para o array
            l = 0; // Reinicia o índice de linhas para os bias
            post("Reading bias of layer %d", m + 1);
            continue;
        }
    
        if (m >= 0 && l < (int)x->net_param[m + 2]) { // Verifica se o índice da camada é válido e se o índice da linha é menor que o nº de neurônios da camada
            char *ptr = line; // Ponteiro para a linha atual
            for (int i = 0; i < (int)x->net_param[m + 2]; i++) { // Percorre os neurônios da camada
                if (sscanf(ptr, "%lf", &x->vetor_bias[m][i]) == 1) { // Lê o valor do bias
                    ptr = strchr(ptr, ' '); // Avança para o próximo número
                    if (!ptr) break;       // Para se não houver mais números
                    ptr++;                 // Move para o próximo caractere após o espaço
                }
            }
            l++; // Avança para a próxima linha de bias
        }
    }    
    fclose(fd);
    post("Model loaded successfully from %s", buf);
}

static void *mlperceptron_new(t_symbol *s, int argc, t_atom *argv) {
    t_mlperceptron *x = (t_mlperceptron *)pd_new(mlperceptron_class);

    //nº de vetores = nº de camadas (matrizes) da rede sem considerar a camada de entrada [vetor de bias, vetor de ativação e vetor de soma ponderada]
    x->num_vetores = 3; 
    x->max_epochs = 100; //nº máximo de épocas
    x->learn_rate = 0.1; //taxa de aprendizado
    x->trainingmode = 0; //modo de treinamento
    x->evalmode = 0; //modo de avaliação
    x->datasize = 10; //tamanho do conjunto de dados
    x->current_data = 0; //nº de dados atuais

    //define os ponteiros como null para evitar ponteiros que apontam para um local de memória inválido ou desalocado
    x->vetor_bias = NULL;
    x->vetor_z = NULL;
    x->vetor_a = NULL;
    x->delta = NULL; 
    x->net_param = NULL;
    x->input_data = NULL;
    x->x_mse = NULL;
    x->x_bias_out = NULL;
    x->class_erro = NULL;
    

    // definição de parâmetros para inicialização 
    x->param = 5; //nº de argumentos recebidos

    // Aloca memória para `net_param`
    x->net_param = (t_float *)malloc(x->param * sizeof(t_float));
    if (!x->net_param) {
        error("Error allocating memory for parameters vector");
        return NULL;
    }

    // Preenche os parâmetros com valores padrão
    x->net_param[0] = 3; //nº de camadas
    x->net_param[1] = 2; //dimensão dos dados de entrada
    x->net_param[2] = 4; //nº de neurônios na primeira camada oculta
    x->net_param[3] = 3; //nº de neurônios na segunda camada oculta
    x->net_param[4] = 2; //nº de neurônios na camada de saída


    // Aloca memória para o array dos dados de entrada
    x->input_data = (double *)malloc(x->net_param[1] * sizeof(double));
    if (!x->input_data) {
        error("Error allocating memory for input data vector.");
        return NULL;
    }
    // post("input data size: %d", (int)x->net_param[1]);


    // Aloca memória para o array das classes dos dados de entrada (tamanh0 = nº de neurônios na camada de saída)
    x->classes = (double *)malloc(x->net_param[x->param-1] * sizeof(double));
    if (!x->classes) {
        error("Error allocating memory for data classes.");
        return NULL;
    }
    // post("Classes: %d", (int)x->net_param[x->param-1]);


    // Aloca memória para o array dos erros de cada classe (tamanho = número de neurônios na camada de saída)
    x->class_erro = (t_float *)malloc(x->net_param[x->param-1] * sizeof(t_float));
    if (!x->class_erro) {
        error("Error allocating memory for error by classes.");
        return NULL;
    }
    

    post("mlperceptron v0.02 (08/01/25) created successfully.");
    
    //atribui o primeiro parâmetro (nº de camadas) como nº de matrizes 
    int num_matrizes = (int)x->net_param[0]; 

    // Libera memória das matrizes antes de realocar
    liberar_hidden_layers(x);
    // Aloca a matriz com as dimensões padrão
    if (!alocar_matrizes(x, num_matrizes)) {
        error("Error allocating memory for hidden layers. Allocation failed.");
        free(x->net_param); // Libera memória para evitar vazamentos
        return NULL;
    }

    //libera memória dos vetores de bias
    liberar_vetores(x, &x->vetor_bias);
    // Aloca os vetores com os tamanhos padrão
    if (!alocar_vetores(x, &x->vetor_bias, num_matrizes)){
        error("Error resizing bias vector. Allocation failed.");
        free(x->net_param); // Libera memória para evitar vazamentos
        liberar_hidden_layers(x);
        return NULL;
    }

    //libera memória dos vetores z 
    liberar_vetores(x, &x->vetor_z);
    // Aloca os vetores z com os tamanhos padrão
    if (!alocar_vetores(x, &x->vetor_z, num_matrizes)){
        error("Error resizing z vector. Allocation failed.");
        free(x->net_param); // Libera memória para evitar vazamentos
        liberar_hidden_layers(x);
        liberar_vetores(x, &x->vetor_bias);
        return NULL;
    }


    //libera memória dos vetores de ativação
    liberar_vetores(x, &x->vetor_a);
    // Aloca os vetores de ativação com os tamanhos padrão
    if (!alocar_vetores(x, &x->vetor_a, num_matrizes)){
        error("Error resizing activation vector. Allocation failed.");
        free(x->net_param); // Libera memória para evitar vazamentos
        liberar_hidden_layers(x);
        liberar_vetores(x, &x->vetor_bias);
        liberar_vetores(x, &x->vetor_z);
        return NULL;
    }

    //libera memória dos vetores de gradientes
    liberar_vetores(x, &x->delta);
    // Aloca os vetores de gradientes com os tamanhos padrão    
    if(!alocar_vetores(x, &x->delta, num_matrizes)){
        error("Error resizing delta vector. Allocation failed.");
        free(x->net_param); // Libera a memória alocada se o array não for null
        liberar_hidden_layers(x);
        liberar_vetores(x, &x->vetor_bias);
        liberar_vetores(x, &x->vetor_z);
        liberar_vetores(x, &x->vetor_a);
        return NULL;
    }

    // cria os outlets
    x->x_out1 = outlet_new(&x->x_obj, &s_list);
    x->x_out2 = outlet_new(&x->x_obj, &s_anything);
    

    // preenche as matrizes e vetores com valores aleatórios entre 0 e 1
    if (x->vetor_bias && x->hidden_layers) {
        hidden_fill(x, 0, 1);
        vetor_fill(x, &x->vetor_bias, 0, 1);
    } else {
        error("Error filling hidden layers and bias vector with random values.");
    } 

    // Aloca memória para o buffer x->x_layer_out com o tamanho do nº de neurônios da camada de saída
    if (x->x_layer_out != NULL) {
        freebytes(x->x_layer_out, x->net_param[x->param-1] * sizeof(t_atom));
        x->x_layer_out = NULL;
        }
    // Aloca memória para o buffer de saída
    x->x_layer_out = (t_atom *)getbytes(x->net_param[x->param-1] * sizeof(t_atom)); 
    // post("Output buffer size: %d", (int)x->net_param[x->param-1]);

    // Aloca memória para o buffer de saída do erro médio quadrático
    if(x->x_mse != NULL){
        freebytes(x->x_mse, sizeof(t_atom));
        x->x_mse = NULL;
    }
    x->x_mse = (t_atom *)getbytes(1 * sizeof(t_atom));

    if(x->x_bias_out != NULL){
        freebytes(x->x_bias_out, sizeof(t_atom));
        x->x_bias_out = NULL;
    }

    // Inicializa o array de funções de ativação
    if(x->activation_function != NULL){
        freebytes(x->activation_function, x->num_hidden * sizeof(t_symbol *));
        x->activation_function = NULL;
    }
    x->activation_function = (t_symbol **)getbytes(x->num_hidden * sizeof(t_symbol *));
    for (int i = 0; i < x->num_hidden; i++) {
        x->activation_function[i] = gensym("sigmoid");  // Função de ativação padrão para todas as camadas
        // post("Activation function %d: %s", i, x->activation_function[i]->s_name);
    }

    x->x_canvas = canvas_getcurrent();

    //inicializa o contador de epócas como zero
    x->current_epoch = 0; //época atual

    post("Network initialized:");
    post("Input data dimension: %d", (int)x->net_param[1]);
    post("Layer 1: %d", (int)x->net_param[2]);
    post("Layer 2: %d", (int)x->net_param[3]);
    post("Output layer: %d", (int)x->net_param[4]);
    post("Default activation function: sigmoid");
    post("Learning rate: %0.2f", x->learn_rate);
    post("Max epochs: %d", x->max_epochs);
    post("Training mode: ON");
    post("Amount of training examples: %d", x->datasize);

    return (void *)x;
}



void mlperceptron_destroy(t_mlperceptron *x) {
    outlet_free(x->x_out1); // Libera o outlet 1
    outlet_free(x->x_out2);// Libera o outlet 2
    liberar_hidden_layers(x);// Libera a memória das matrizes
    liberar_vetores(x, &x->vetor_bias);// Libera a memória dos vetores
    liberar_vetores(x, &x->vetor_z);// Libera a memória dos vetores z
    liberar_vetores(x, &x->vetor_a);// Libera a memória dos vetores de ativação
    liberar_vetores(x, &x->delta);// Libera a memória dos vetores de gradientes 
    
    if (x->x_linha_out != NULL) {
        freebytes(x->x_linha_out, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->x_linha_out = NULL;
    }
    if (x->x_layer_out != NULL) {
        freebytes(x->x_layer_out, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->x_layer_out = NULL;
    }
    if(x->x_mse != NULL){
        freebytes(x->x_mse, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->x_mse = NULL;
    }
    if(x->x_bias_out != NULL){
        freebytes(x->x_bias_out, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->x_bias_out = NULL;
    }
    if (x->net_param != NULL) {
        free(x->net_param);  // Libera a memória do array de parâmetros
        x->net_param = NULL;
    }
    if (x->input_data != NULL) {
        free(x->input_data);  // Libera a memória do array de parâmetros
        x->input_data = NULL;
    }
    if (x->classes != NULL) {
        free(x->classes);  // Libera a memória do array de parâmetros
        x->classes = NULL;
    }
    if (x->class_erro != NULL) {
        free(x->class_erro);  // Libera a memória do array de erros por classe
        x->class_erro = NULL;
    }
    if (x->activation_function != NULL) {
        freebytes(x->activation_function, x->num_hidden * sizeof(t_symbol *)); // Libera a memória do array de funções de ativação
        x->activation_function = NULL;
    }
    
}
void mlperceptron_setup(void) {
    mlperceptron_class = class_new(
        gensym("mlperceptron"), // Nome do objeto
        (t_newmethod)mlperceptron_new, // Chama a função construtor
        (t_method)mlperceptron_destroy, // Chama a função destruidor
        sizeof(t_mlperceptron),
        CLASS_DEFAULT,
        A_DEFFLOAT, 0); // Tamanho do objeto

    class_addlist(mlperceptron_class, (t_method) training_data); //recebe lista de dados de entrada e propaga os dados
    class_addmethod(mlperceptron_class, (t_method)matriz_size, gensym("size"), A_GIMME, 0); //redimensiona as matrizes (camadas escondidas) e vetores (bias, ativação, soma ponderada)
    class_addmethod(mlperceptron_class, (t_method)matrizes_out, gensym("weight"), A_GIMME, 0); //retorna linha por linha de todas matrizes
    class_addmethod(mlperceptron_class, (t_method)random_init, gensym("random"), A_GIMME, 0); //preenche os vetores de bias e matriz de pesos com valores aleatórios
    class_addmethod(mlperceptron_class, (t_method) training, gensym("training"), A_FLOAT, 0); //define o modo de treinamento
    class_addmethod(mlperceptron_class, (t_method) eval_mode, gensym("evaluation"), A_FLOAT, 0); //define o modo de avaliação
    class_addmethod(mlperceptron_class, (t_method) learning_rate, gensym("learning"), A_FLOAT, 0);//define a taxa de aprendizado
    class_addmethod(mlperceptron_class, (t_method) epoch_amount, gensym("epochs"), A_FLOAT, 0);//define o nº máximo de épocas
    class_addmethod(mlperceptron_class, (t_method) datasize, gensym("datasize"), A_FLOAT, 0);//define o nº de exemplos de treinamento
    class_addmethod(mlperceptron_class, (t_method) reset, gensym("reset"), A_GIMME, 0);//reseta os parâmetros da rede para valores padrão (pesos, bias, taxa, épocas)
    class_addmethod(mlperceptron_class, (t_method) vetores_out, gensym("bias"), A_GIMME, 0); //retorna os vetores de bias
    class_addmethod(mlperceptron_class, (t_method) activation_functions, gensym("af"), A_GIMME, 0); //define as funções de ativação para cada camada
    class_addmethod(mlperceptron_class, (t_method) random_uniforme, gensym("uniform"), A_GIMME, 0); //preenche os vetores de bias e matriz de pesos com valores aleatórios uniformes 
    class_addmethod(mlperceptron_class, (t_method) random_he, gensym("he"), A_GIMME, 0); //preenche os vetores de bias e matriz de pesos com valores aleatórios he
    class_addmethod(mlperceptron_class, (t_method) random_lecun, gensym("lecun"), A_GIMME, 0); //preenche os vetores de bias e matriz de pesos com valores aleatórios lecun
    class_addmethod(mlperceptron_class, (t_method) random_xavier, gensym("xavier"), A_GIMME, 0);//preenche os vetores de bias e matriz de pesos com valores aleatórios xavier
    class_addmethod(mlperceptron_class, (t_method) model_save, gensym("write"), A_SYMBOL, 0); //salva o modelo treinado
    class_addmethod(mlperceptron_class, (t_method) model_load, gensym("read"), A_SYMBOL, 0); //carrega o modelo treinado
}
