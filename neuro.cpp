// Модель искусственной нейросети

using namespace std;
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

const int N_LAYER = 3, //число слоев = 3 + нулевой
    N_MAX = 30, //максимально возможное число нейронов в слое
    N_MIN = 10, //минимально возможное число нейронов в слое
    N_PAT = 10, //число шаблонов (patterns)
    LR = 1,   //Коэффициент скорости обучения (leaning rate)
    N_I = 15000; // максимальное количество итераций в цикле обучения
const float ERR_MAX = 0.01; // пороговая ошибка обучения

int struc[N_LAYER + 1] = {N_MAX, 30, 20, N_MIN}; // структура сети
float w[N_LAYER + 1][N_MAX][N_MAX];              //веса
/*
  w[k][2][3]
  k - номер слоя
  2 - номер нейрона в (k-1) слое
  3 - номер нейрона в k-ом слое}
*/
float pattern[N_PAT][N_MAX]; //совокупность шаблонов
// pattern[1][2] - второй пиксел шаблона №1
float target[N_PAT][N_MIN]; //целевой вектор
float outs[N_LAYER + 1][N_MAX]; //Выходные значения нейронов в каждом слое
float delta[N_LAYER + 1][N_MAX]; //сигналы ошибки дельта
float eps;                    // Эпсилон

int c, //вспомогательный счетчик
    m, //номер шаблона
    k, //номер слоя
    j, //номер нейрона в предыдущем слое
    i; //номер нейрона в текущем слое

bool exitFlag;


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
// Функуия вывода прогресс-бара для красоты
void printProgress(double percentage) {
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}


// Проход вперед
void neuroCalc() {
  float net;
  for (k = 1; k <= N_LAYER; k++) {
    for (i = 0; i < struc[k]; i++) {
      net = 0;
      for (j = 0; j < struc[k - 1]; j++) {
        net += outs[k - 1][j] * w[k][j][i];
      }
      outs[k][i] = 1 / (1 + exp(-net));
    }
  }
}


// Расчет ошибки для шаблона m
float calcErr(int m) {
  // Вспомогательная лок. переменная для подсчёта суммы
  float sum = 0;
  for (i = 0; i < N_MIN; i++) {
    sum += pow(target[m][i] - outs[N_LAYER][i], 2);
  }
  return sqrt(sum / N_MIN);
}


// Расчет суммарной ошибки по всем шаблонам
float calcSumErr(void) {
  float sum = 0;
  for (c = 0; c < N_PAT; c++) {
    for (i = 0; i < N_MAX; i++) {
      outs[0][i] = pattern[c][i];
    }

    neuroCalc();

    sum += pow(calcErr(c), 2);
  }
  return sqrt(sum / N_PAT);
}


// Проход назад
void goBack(void) {
  for (k = N_LAYER; k >= 1; k--) {
    for (i = 0; i < struc[k]; i++) {
      if (k == N_LAYER) {
        eps = target[m][i] - outs[k][i];
      }
      else {
        eps = 0;
        for (c = 0; c < struc[k + 1]; c++) {
          eps += delta[k + 1][c] * w[k + 1][i][c];
        }
      }
      delta[k][i] = outs[k][i] * (1 - outs[k][i]) * eps;
      for (j = 0; j < struc[k - 1]; j++) {
        w[k][j][i] += LR * delta[k][i] * outs[k - 1][j];
      }
    }
  }
}


// Обучение
void educate(void) {
  cout << "Education..." << endl;

  int count = 0; // Счётчик итераций
  m = 0;  // Счётчик шаблонов (глобальный)
  float sum = 0;   // Сумма для подсчёта ошибки
  float sigma = 1; // Ошибка по всем шаблонам (ср. квадратическое откл.)

  do {
    // Заполнение нулевого слоя очередным шаблоном
    for (i = 0; i < N_MAX; i++) {
      outs[0][i] = pattern[m][i];
    }
    neuroCalc(); // Вычисление выхода нейросети
    sum += pow(calcErr(m), 2);
    goBack(); // Проход назад
    // Выбираем очередную обучающую пару
    if (m == N_PAT - 1) {
      sigma = sqrt(sum / N_PAT);
      m = 0;
      sum = 0;
    } else
      m++;
    count++;

    // Вывод прогресс-бара каждый сотый шаг
    if (count % 100 == 0) {
      printProgress((float)count / N_I);
    }

  } while (sigma > ERR_MAX && count < N_I);

  cout << "\n";
  cout << "Education is completed for " << count << " iterations." << endl;
}


// Процедура инициализации
void init(void) {
  // Начальная инициализация весов
  srand(time(NULL));
  for (k = 1; k <= N_LAYER; k++)
    for (j = 0; j < struc[k - 1]; j++)
      for (i = 0; i < struc[k]; i++)
        w[k][j][i] = -1 + 2 * (float)rand() / RAND_MAX;
}


// Converts a vector of values to a probability distribution.
// The elements of the output vector are in range [0, 1] and sum to 1.
// The probability of each vector x is computed as `x / sum(x)`
// В Keras есть спец. функция `softmax`
void calcProbability(void) {
  float sum = 0; // для расчёта суммы
  float prob = 0; // значение вероятности (нормированное значение)

  float maxIndex = 0; // индекс максимального элемента массива
  float maxValue = 0; // максимальное значение в массиве

  // Расчёт суммы sum(exp(x))
  for (i = 0; i < N_MIN; i++) {
//    sum += exp(outs[N_LAYER][i]);
    sum += outs[N_LAYER][i];
  }

  cout << "Probability:" << endl;
  for (i = 0; i < N_MIN; i++) {
    //prob = exp(outs[N_LAYER][i]) / sum;
    prob = outs[N_LAYER][i] / sum;
    cout << setprecision(2) << prob * 100 << "  ";

    if (maxValue < prob) {
      maxValue = prob;
      maxIndex = i;
    }
  }

  cout << "\n";

  cout << "Prediction label: " << maxIndex << endl;
}


// Вывод на экран нейросети: вход, цель, выход, ошибка
// Вызывается после процедуры neuroCalc()
void printNeuronet(void) {
  //Вывод входного изображения
  cout << "\n";
  cout << "Input: " << endl;
  for (i = 0; i < N_MAX; i++) {
    if (outs[0][i] == 1)
      cout << '#';
    else
      cout << '.';
    // После каждого пятого символа переводим строку
    if ((i + 1) % 5 == 0)
      cout << '\n';
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
    cout << setprecision(2) << outs[N_LAYER][i] << "  ";
  }
  cout << "\n\n";

  //Вывод ошибки
  cout << "Err: " << setprecision(4) << calcErr(m) << endl;
  cout << "\n";

  calcProbability();
  cout << "\n\n";
}


// Процедура загрузки шаблонов из файла
void loadPatterns(void) {
  cout << "Loading patterns..." << endl;
  ifstream f;
  f.clear();
  f.open("patterns.txt");
  if (!f) {
    cout << "Can't open patterns.txt";
  } else {
    while (!f.eof()) {
      f >> m; //считывание номера шаблона
      cout << "m = " << m << endl;
      // Задаём целевой вектор - 1 в номере шаблона
      target[m][m] = 1;
      // считываем входной шаблон
      cout << "Pattern: ";
      for (i = 0; i < N_MAX; i++) {
        f >> pattern[m][i];
        cout << pattern[m][i] << ' ';
      }
      cout << "\n\n";
    }
    f.close();
    cout << "Patterns are loaded!" << endl;
  }
}


// Загрузка изображения из файла
void loadInput() {
  cout << "Load <input.txt>...\n" << endl;

  // считываем входное изображение
  ifstream f;
  f.clear();
  f.open("input.txt");
  if (!f) {
    cout << "Can't open input.txt";
  } else {
    f >> m; //считывание истинного номера шаблона
    cout << "True label: " << m << endl;
    // считываем входное изображение
    for (i = 0; i < N_MAX; i++)
      f >> outs[0][i];
    f.close();
  }
}


// Расчёт шаблона
void calcPattern() {
  cout << "Calc pattern..." << endl;
  cout << "Enter pattern number: " << endl;
  cin >> m;

  // Заполнение нулевого слоя шаблоном m
  for (i = 0; i < N_MAX; i++)
    outs[0][i] = pattern[m][i];

  neuroCalc();

  printNeuronet();
}


// Расчёт изображения из файла <input.txt>
void calcInput() {
  cout << "Calc input..." << endl;

  loadInput();

  neuroCalc();

  printNeuronet();
}


// Изменение структуры нейросети
void setStruc(void) {
  cout << "Set structure..." << endl;
  cout << "Enter N1 (30) and N2 (25): "
       << "\n";
  cin >> struc[1] >> struc[2];
  cout << "Sum err: " << calcSumErr() << endl;
}


// Обработчик выхода из программы
void exitProgram(void) {
  char ch = 'n';

  do {
    cout << "Exit anyway? (y or n)" << endl;
    cin >> ch;
  }
  while (ch != 'y' && ch != 'n');

  if (ch == 'y') exitFlag = true;
  if (ch == 'n') exitFlag = false;
}


// Основная функция
int main(void) {
  init(); //инициализация весов

  loadPatterns(); //Загрузка шаблонов

  exitFlag = false; //сброс флага выхода

  char ch; //код команды
  // Вывод меню на экран
  do {
    cout << "\nMenu:" << endl;
    cout << "1: Load patterns" << endl;
    cout << "2: Calc pattern" << endl;
    cout << "3: Educate" << endl;
    cout << "4: Calc input" << endl;
    cout << "5: Set structure" << endl;
    cout << "6: Exit" << endl;
    cout << "\n   Select menu item: ";
    cin >> ch;
    cout << endl;

    switch (ch) {
    case '1':
      loadPatterns();
      break;
    case '2':
      calcPattern();
      break;
    case '3':
      educate();
      break;
    case '4':
      calcInput();
      break;
    case '5':
      setStruc();
      break;
    case '6':
      exitProgram();
      break;
    default:
      cout << "Wrong item! Try again!" << endl;
      break;
    }
  } while (!exitFlag);

  cout << "The program is closed. Goodbye!" << endl;

  return 0;
}

