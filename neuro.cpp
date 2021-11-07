using namespace std;
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>

const int N_SLOY = 3, //число слоев = 3 + нулевой
  N_MAX = 30, //максимально возможное число нейронов в слое
  N_PATTERN = 10, //число шаблонов
  KSO = 1; //Коэффициент скорости обучения
const float CONST_ERROR = 0.01;
const int STRUC_CONST[N_SLOY+1] = {30,30,25,4}; //чило нейронов в каждом слое

int struc[N_SLOY+1] = {30,30,25,4};
float w[N_SLOY+1][N_MAX][N_MAX]; //веса
/*
  w[k][2][3]
  k - номер слоя
  2 - номер нейрона в (k-1) слое
  3 - номер нейрона в k-ом слое}
*/
float pattern[N_PATTERN][N_MAX]; //совокупность шаблонов
//pattern[1][2] - второй пиксел шаблона №1
float target[N_PATTERN][4]; //целевой вектор
float outArr[N_SLOY+1][N_MAX]; //Выходные значения нейронов в каждом слое
float delta[N_SLOY+1][N_MAX];   //сигналы ошибки
float dw; //поправка для веса;
float error;  //ошибка;

int c, //вспомогательный счетчик
  m, //номер шаблона
  k, //номер слоя
  j, //номер нейрона в предыдущем слое
  i, //номер нейрона в текущем слое
  count_error; //счетчик итераций при обучении

float error1; //ошибка-критерий остановки обучения

void forwardPass()
{
  for (k = 1; k <= N_SLOY; k++)
    for (i = 0; i < struc[k]; i++) {
      outArr[k][i] = 0;
      for (j = 0; j < struc[k - 1]; j++)
        outArr[k][i] = outArr[k][i] + outArr[k-1][j] * w[k][j][i];
      outArr[k][i] = 1/(1+exp(-outArr[k][i]));
    }
}

float calcErr(int m)
{
  for (i = 0; i < struc[0]; i++) {
    outArr[0][i] = pattern[m][i];
  }
  forwardPass();
  float err = 0;
  for (i = 0; i < struc[N_SLOY]; i++) {
    err += pow(target[m][i] - outArr[N_SLOY][i], 2);
  }
  return sqrt(err/struc[N_SLOY]);
}

float calcSumErr(void)
{
  float err = 0;
  for (int m = 0; m < N_PATTERN; m++) {
    err += pow(calcErr(m), 2);
  }
  return sqrt(err/N_PATTERN);
}

void backPropagation(void)
{
  count_error = 0;
  m = 0;
  do {
    for (i = 0; i < struc[0]; i++) {
      outArr[0][i] = pattern[m][i];
    }
    forwardPass();
    for (k = N_SLOY; k >= 1; k--)
      for (i = 0; i < struc[k]; i++) {
        if (k == N_SLOY)
          error = target[m][i] - outArr[k][i];
        else {
          error = 0;
          for (c = 0; c < struc[k + 1]; c++)
            error += delta[k + 1][c] * w[k + 1][i][c];
        }
        delta[k][i] = outArr[k][i] * (1 - outArr[k][i]) * error;
        for (j = 0; j < struc[k - 1]; j++) {
          dw = KSO * delta[k][i] * outArr[k - 1][j];
          w[k][j][i] += dw;
        }    
      }
    if (m == N_PATTERN - 1) m = 0;
    else m++;
    count_error++;
  }
  while ((count_error < 20000) and (calcSumErr() > CONST_ERROR));
  cout << "Education is completed for " << count_error << " iterations." << endl ;
	//Education is completed for 2 iterations
}

void init(void)
{
  // Начальная инициализация весов
  srand(time(NULL));
  for (k = 1; k <= N_SLOY; k++)
    for (j = 0; j < struc[k - 1]; j++)
      for (i = 0; i < struc[k]; i++)
        w[k][j][i] = -1 + 2 * (float)rand()/RAND_MAX;
}

void writeArr(float *arr, int size, int car = 5)
{
  for (i = 0; i < size; i++) {
    cout << arr[i] << ' ';
    if ((i + 1) % car == 0) cout << '\n'; 
  }
  cout << '\n';
}

void loadPatterns(void)
{
  cout << "Loading patterns..." << endl;
  ifstream f;
  f.clear();
  f.open("patterns.txt");
  if (!f) {
    cout << "Can't open patterns.txt";
  }
  else {
    while (!f.eof()) {
      f >> m; //считывание номера шаблона
      // считываем входной шаблон
      for (i = 0; i < struc[0]; i++) {
        f >> pattern[m][i];
      }
      // считываем цель
      for (i = 0; i < struc[N_SLOY]; i++) {
        f >> target[m][i];
      }
    }
    f.close();
    cout << "Patterns are loaded!" << endl;
  }  
}

void viewPattern(void)
{
  cout << "Type pattern num: ";
  cin >> m;  cout << "Pattern #" << m << ": " << endl;
  writeArr(pattern[m], struc[0]);
  cout << "Target #" << m << ": " << endl;
  writeArr(target[m], struc[N_SLOY]);  
}

void calcOut(void)
{
  cout << "Calc output..." << endl;

  // считываем входное изображение
  ifstream f;
  f.clear();
  f.open("input.txt");
  if (!f) {
    cout << "Can't open input.txt";
  }
  else {
    while (!f.eof()) {
      f >> m; //считывание номера шаблона
      // считываем входное изображение
      for (i = 0; i < struc[0]; i++) {
        f >> outArr[0][i];
      }
      // считываем цель
      //for (i = 0; i < struc[N_SLOY]; i++) {
      //  f >> target[m][i];
      //}
    }
    f.close();   
  }
  
  cout << "Input: " << endl;
  writeArr(outArr[0], struc[0]);
  cout << "Target #" << m << ": " << endl;
  writeArr(target[m], struc[N_SLOY]);
  
  forwardPass();
  
  cout << "Output: " << endl;
  writeArr(outArr[N_SLOY], struc[N_SLOY]);

  cout << "Err: " << calcErr(m) << endl;
}

int main(void)
{
  init(); //инициализация весов
  
  char ch; //код команды
  do {
    cout << "\nMenu:" << endl;
    cout << "1: Load patterns" << endl;
    cout << "2: Calc out" << endl;
    cout << "3: Educate" << endl;
    cout << "4: Set structure" << endl;
    cout << "5: Calc sum Err" << endl;
    cout << "6: View pattern" << endl;
    cout << "7: Exit" << endl;
    cout << "\n   Select menu item: ";
    cin >> ch;
    
    switch(ch) {
    case '1':
      loadPatterns();
      break;
    case '2':
      calcOut();
      break;
    case '3':
      backPropagation();
      break;
    case '4':
      struc[1] = 30;
      struc[2] = 25;
      cout << "Sum err: " << calcSumErr() << endl;      
      break;
    case '5':
      cout << "Sum err: " << calcSumErr() << endl;
      break;    
    case '6':
      viewPattern();
      break;    
    }
  } while (ch != '7');
  
  //system("pause");
  
  return 0;
}
