using namespace std;
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

const int N_SLOY = 3, //число слоев = 3 + нулевой
  N_MAX = 30, //максимально возможное число нейронов в слое
  N_MIN = 4, //минимально возможное число нейронов в слое
  N_PATTERN = 10, //число шаблонов
  KSO = 1; //Коэффициент скорости обучения
const float CONST_ERROR = 0.01;

int struc[N_SLOY + 1] = {N_MAX,30,25,N_MIN};
float w[N_SLOY + 1][N_MAX][N_MAX]; //веса
/*
  w[k][2][3]
  k - номер слоя
  2 - номер нейрона в (k-1) слое
  3 - номер нейрона в k-ом слое}
*/
float pattern[N_PATTERN][N_MAX]; //совокупность шаблонов
//pattern[1][2] - второй пиксел шаблона №1
float target[N_PATTERN][N_MIN]; //целевой вектор
float outs[N_SLOY + 1][N_MAX]; //Выходные значения нейронов в каждом слое
float delta[N_SLOY + 1][N_MAX];   //сигналы ошибки
float dw; //поправка для веса;
float error;  //ошибка;

int c, //вспомогательный счетчик
  m, //номер шаблона
  k, //номер слоя
  j, //номер нейрона в предыдущем слое
  i, //номер нейрона в текущем слое
  count_error; //счетчик итераций при обучении

void neuroCalc()
{
  for (k = 1; k <= N_SLOY; k++)
    for (i = 0; i < struc[k]; i++) {
      outs[k][i] = 0;
      for (j = 0; j < struc[k - 1]; j++)
        outs[k][i] = outs[k][i] + outs[k-1][j] * w[k][j][i];
      outs[k][i] = 1/(1+exp(-outs[k][i]));
    }
}

float calcErr(int m)
{
  float err = 0;
  for (i = 0; i < N_MIN; i++) {
    err += pow(target[m][i] - outs[N_SLOY][i], 2);
  }
  return sqrt(err/N_MIN);
}

float calcSumErr(void)
{
  float err = 0;
  for (c = 0; c < N_PATTERN; c++) {
    for (i = 0; i < struc[0]; i++) {
      outs[0][i] = pattern[c][i];
    }
    
    neuroCalc();

    err += pow(calcErr(c), 2);
  }
  return sqrt(err/N_PATTERN);
}

void goBack(void)
{
  for (k = N_SLOY; k >= 1; k--)
    for (i = 0; i < struc[k]; i++) {
      if (k == N_SLOY)
        error = target[m][i] - outs[k][i];
      else {
        error = 0;
        for (c = 0; c < struc[k + 1]; c++)
          error += delta[k + 1][c] * w[k + 1][i][c];
      }
      delta[k][i] = outs[k][i] * (1 - outs[k][i]) * error;
      for (j = 0; j < struc[k - 1]; j++) {
        dw = KSO * delta[k][i] * outs[k - 1][j];
        w[k][j][i] += dw;
      }    
    }
}

void educate(void)
{
  count_error = 0;
  m = 0;
  do {
    for (i = 0; i < struc[0]; i++) {
      outs[0][i] = pattern[m][i];
    }
    neuroCalc();
    goBack();
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
    cout << outs[N_SLOY][i] << ' ';
  }
  cout << "\n\n";  
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

void calcPattern(void)
{
  cout << "Calc pattern" << endl;

  cout << "Type pattern num: ";
  cin >> m;

  for (i = 0; i < N_MAX; i++) {
    outs[0][i] = pattern[m][i];
  }

  neuroCalc();
  
  printNeuronet();
  
  cout << "Err: " << calcErr(m) << endl;  
}

void calcInput(void)
{
  cout << "Calc input.txt" << endl;

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
    cout << "2: Calc pattern" << endl;
    cout << "3: Calc input" << endl;
    cout << "4: Educate" << endl;
    cout << "5: Set structure" << endl;
    cout << "6: Exit" << endl;
    cout << "\n   Select menu item: ";
    cin >> ch;
    
    switch(ch) {
    case '1':
      loadPatterns();
      break;
    case '2':
      calcPattern();;
      break;
    case '3':
      calcInput();
      break;
    case '4':
      educate();
      break;
    case '5':
      setStruc();
      break;
    }
  } while (ch != '6');
  
  //system("pause");
  
  return 0;
}
