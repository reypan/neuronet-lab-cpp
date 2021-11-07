// Модель искусственной нейросети

using namespace std;
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <ctime>

const int N_SL = 3, //число слоев = 3 + нулевой
  N_MAX = 30, //максимально возможное число нейронов в слое
  N_MIN = 4, //минимально возможное число нейронов в слое
  N_SH = 10, //число шаблонов
  KSO = 1, //Коэффициент скорости обучения
  N_I = 15000; // максимальное количество итераций в цикле обучения
const float E_P = 0.01; // пороговая ошибка обучения

int struc[N_SL + 1] = {N_MAX, 30, 25, N_MIN}; // структура сети
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
float delta[N_SL + 1][N_MAX]; //сигналы ошибки дельта
float eps; // Эпсилон

int c, //вспомогательный счетчик
  m, //номер шаблона
  k, //номер слоя
  j, //номер нейрона в предыдущем слое
  i; //номер нейрона в текущем слое


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
// Функуия вывода прогресс-бара для красоты
void printProgress(double percentage) {
  int val = (int) (percentage * 100);
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}


// Проход вперед
void neuroCalc()
{
  float net;
  for (k = 1; k <= N_SL; k++)
    for (i = 0; i < struc[k]; i++) {
      net = 0;
      for (j = 0; j < struc[k - 1]; j++)
        net += outs[k-1][j] * w[k][j][i];
      outs[k][i] = 1/(1+exp(-net));
    }
}


// Расчет ошибки для шаблона m
float calcErr(int m)
{
  // Вспомогательная лок. переменная для подсчёта суммы
  float sum = 0;
  for (i = 0; i < N_MIN; i++) {
    sum += pow(target[m][i] - outs[N_SL][i], 2);
  }
  return sqrt(sum/N_MIN);
}


// Расчет суммарной ошибки по всем шаблонам
float calcSumErr(void)
{
  float sum = 0;
  for (c = 0; c < N_SH; c++) {
    for (i = 0; i < N_MAX; i++) {
      outs[0][i] = pattern[c][i];
    }
    
    neuroCalc();

    sum += pow(calcErr(c), 2);
  }
  return sqrt(sum/N_SH);
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
  cout << "Education..." << endl;
  
  int count = 0;  // Счётчик итераций
  m = 0;
  float sum = 0;  // Сумма для подсчёта ошибки 
  float sigma = 1;  // Ошибка по всем шаблонам (ср. квадратическое откл.)

  do {
    // Заполнение нулевого слоя очередным шаблоном
    for (i = 0; i < N_MAX; i++) {
      outs[0][i] = pattern[m][i];
    }
    neuroCalc();                // Вычисление выхода нейросети
    sum += pow(calcErr(m), 2);
    goBack();                   // Проход назад
    // Выбираем очередную обучающую пару
    if (m == N_SH - 1) {
      sigma = sqrt(sum/N_SH);
      m = 0;
      sum = 0;
    }
    else m++;
    count++;

    // Вывод прогресс-бара каждый сотый шаг
    if (count % 100 == 0)
      printProgress((float)count / N_I);
    
  }
  while ( sigma > E_P && count < N_I );

  cout << "\n";
  cout << "Education is completed for " << count << " iterations." << endl;
}


// Процедура инициализации
void init(void)
{
  // Начальная инициализация весов
  srand(time(NULL));
  for (k = 1; k <= N_SL; k++)
    for (j = 0; j < struc[k - 1]; j++)
      for (i = 0; i < struc[k]; i++)
        w[k][j][i] = -1 + 2 * (float)rand() / RAND_MAX;
}


// Вывод на экран нейросети: вход, цель, выход, ошибка
// Вызывается после процедуры neuroCalc()
void printNeuronet(void)
{
  //Вывод входного изображения
  cout << "\n";
  cout << "Input: " << endl;
  for (i = 0; i < N_MAX; i++) {
    if (outs[0][i] == 1) cout << '#';
    else cout << '.';
    // После каждого пятого символа переводим строку
    if ((i + 1) % 5 == 0) cout << '\n'; 
  }
  cout << "\n";

  //Вывод целевого вектора
  cout << "Target: " << endl;
  for (i = 0; i < N_MIN; i++) {
    cout << target[m][i] << "  ";
  }
  cout << "\n\n";

  //Вывод выходного вектора
  cout << "Output: " << endl;
  for (i = 0; i < N_MIN; i++) {
    cout << setprecision(2) << outs[N_SL][i] << "  "; 
  }
  cout << "\n\n";

  //Вывод ошибки
  cout << "Err: " << setprecision(4) << calcErr(m) << endl;  
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
  else loadInput();  // Загрузка из файла

  neuroCalc();
  
  printNeuronet();
}

void setStruc(void)
{
  cout << "Set structure..." << endl;
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
