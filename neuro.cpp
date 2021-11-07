using namespace std;
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

const int N_SL = 3, //число слоев = 3 + нулевой
  N_MAX = 30, //максимально возможное число нейронов в слое
  N_MIN = 4, //минимально возможное число нейронов в слое
  N_SH = 10, //число шаблонов
  KSO = 1, //Коэффициент скорости обучения
  N_I = 20000; // максимальное количество итераций в цикле обучения
const float E_P = 0.01;

int struc[N_SL + 1] = {N_MAX,30,25,N_MIN};
float w[N_SL + 1][N_MAX][N_MAX]; //веса
/*
  w[k][2][3]
  k - номер слоя
  2 - номер нейрона в (k-1) слое
  3 - номер нейрона в k-ом слое}
*/
float pattern[N_SH][N_MAX]; //совокупность шаблонов
//pattern[1][2] - второй пиксел шаблона №1
float target[N_SH][N_MIN]; //целевой вектор
float outs[N_SL + 1][N_MAX]; //Выходные значения нейронов в каждом слое
float delta[N_SL + 1][N_MAX];   //сигналы ошибки
float eps;  // Эпсилон;

int c, //вспомогательный счетчик
  m, //номер шаблона
  k, //номер слоя
  j, //номер нейрона в предыдущем слое
  i, //номер нейрона в текущем слое
  count_error; //счетчик итераций при обучении

// Проход вперед
void neuroCalc()
{
  for (k = 1; k <= N_SL; k++)
    for (i = 0; i < struc[k]; i++) {
      outs[k][i] = 0;
      for (j = 0; j < struc[k - 1]; j++)
        outs[k][i] += outs[k-1][j] * w[k][j][i];
      outs[k][i] = 1/(1+exp(-outs[k][i]));
    }
}

// Расчет ошибки для шаблона m
float calcErr(int m)
{
  float err = 0;
  for (i = 0; i < N_MIN; i++) {
    err += pow(target[m][i] - outs[N_SL][i], 2);
  }
  return sqrt(err/N_MIN);
}

// Расчет суммарной ошибки
float calcSumErr(void)
{
  float err = 0;
  for (c = 0; c < N_SH; c++) {
    for (i = 0; i < N_MAX; i++) {
      outs[0][i] = pattern[c][i];
    }
    
    neuroCalc();

    err += pow(calcErr(c), 2);
  }
  return sqrt(err/N_SH);
}

// Проход назад
void goBack(void)
{
  for (k = N_SL; k >= 1; k--)
    for (i = 0; i < struc[k]; i++) {
      if (k == N_SL)
        eps = target[m][i] - outs[k][i];
      else {
        eps = 0;
        for (c = 0; c < struc[k + 1]; c++)
          eps += delta[k + 1][c] * w[k + 1][i][c];
      }
      delta[k][i] = outs[k][i] * (1 - outs[k][i]) * eps;
      for (j = 0; j < struc[k - 1]; j++) {
        w[k][j][i] += KSO * delta[k][i] * outs[k - 1][j];
      }    
    }
}

// Обучение
void educate(void)
{
  count_error = 0;
  m = 0;
  do {
    // Заполнение нулевого слоя очередным шаблоном
    for (i = 0; i < N_MAX; i++) {
      outs[0][i] = pattern[m][i];
    }
    neuroCalc();                // Вычисление выхода нейросети
    goBack();                   // Проход назад
    // Выбираем очередную обучающую пару
    if (m == N_SH - 1) m = 0;
    else m++;
    count_error++;
  }
  while ((count_error < N_I) and (calcSumErr() > E_P));
  cout << "Education is completed for " << count_error << " iterations." << endl ;
  // Education is completed for 2 iterations
}

// Процедура инициализации
void init(void)
{
  // Начальная инициализация весов
  srand(time(NULL));
  for (k = 1; k <= N_SL; k++)
    for (j = 0; j < struc[k - 1]; j++)
      for (i = 0; i < struc[k]; i++)
        w[k][j][i] = -1 + 2 * (float)rand()/RAND_MAX;
}

//Вывод на экран нейросети
void printNeuronet(void)
{
  //Вывод входного изображения
  cout << "\n";
  cout << "Input: " << endl;
  for (i = 0; i < N_MAX; i++) {
    if (outs[0][i] == 1) cout << char(219);
    else cout << char(176);
    if ((i + 1) % 5 == 0) cout << '\n'; 
  }
  cout << "\n";

  //Вывод целевого вектора
  cout << "Target: " << endl;
  for (i = 0; i < N_MIN; i++) {
    cout << target[m][i] << ' ';
  }
  cout << "\n\n";

  //Вывод выходного вектора
  cout << "Output: " << endl;
  for (i = 0; i < N_MIN; i++) {
    cout << outs[N_SL][i] << ' ';
  }
  cout << "\n\n";  
}

// Процедура загрузки шаблонов из файла
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
      cout << "m = " <<  m << endl;
      // считываем входной шаблон
      cout << "Pattern: ";
      for (i = 0; i < N_MAX; i++) {
        f >> pattern[m][i];
        cout << pattern[m][i] << ' ';        
      }
      cout << "\n";
      // считываем цель
      cout << "Target: ";
      for (i = 0; i < N_MIN; i++) {
        f >> target[m][i];
        cout << target[m][i] << ' ';
      }
      cout << "\n\n";
    }
    f.close();
    cout << "Patterns are loaded!" << endl;
  }  
}

// Загрузка изображения из файла
void loadInput()
{
  cout << "Load <input.txt>..." << endl;

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
      for (i = 0; i < N_MAX; i++) {
        f >> outs[0][i];
      }
    }
    f.close();   
  }  
}

void calcOutput()
{
  cout << "Calc output..." << endl;
  cout << "Enter pattern number from 0 to 9 or 10 for load input.txt: " << endl;

  cout << "Type pattern num: ";
  cin >> m;

  if (m >= 0 && m <= 9) {
    // Заполнение нулевого слоя шаблоном m
    for (i = 0; i < N_MAX; i++) 
      outs[0][i] = pattern[m][i];
  }
  else loadInput();               // Загрузка из файла

  neuroCalc();
  
  printNeuronet();
  
  cout << "Err: " << calcErr(m) << endl;  
}

void setStruc(void)
{
  cout << "Enter N1 (30) and N2 (25): " << "\n";
  cin >> struc[1] >> struc[2];
  cout << "Sum err: " << calcSumErr() << endl;  
}


int main(void)
{
  init(); //инициализация весов

  loadPatterns(); //Загрузка шаблонов
  
  char ch; //код команды
  do {
    cout << "\nMenu:" << endl;
    cout << "1: Load patterns" << endl;
    cout << "2: Calc output" << endl;
    cout << "3: Educate" << endl;
    cout << "4: Set structure" << endl;
    cout << "5: Exit" << endl;
    cout << "\n   Select menu item: ";
    cin >> ch;
    
    switch(ch) {
    case '1':
      loadPatterns();
      break;
    case '2':
      calcOutput();
      break;
    case '3':
      educate();
      break;
    case '4':
      setStruc();
      break;
    }
  } while (ch != '5');
  
  //system("pause");
  
  return 0;
}
