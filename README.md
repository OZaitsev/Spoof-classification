<h1 style="text-align: center;">Spoof classification</h1> 
<h2 style="text-align: center;">Описание задачи</h2>

В настоящее время для решения навигационной задачи в современной технике активно развивается направление, основанное на интеграции инерциальных и спутниковых навигационных систем. Интеграция способствует более эффективному решению навигационной задачи и предотвращению сбоев, поскольку позволяет использовать достоинства каждой из систем и компенсировать недостатки, присущие другой системе.  ИНС характеризуется низким уровнем шумовой составляющей погрешности измерения и высокой частотой обновления данных, в то же время погрешность измерения носит нестационарный характер, т. е. имеет место накапливающаяся ошибка. ГНСС, наоборот, характеризуется отсутствием накапливающихся ошибок, но высоким уровнем шумовой составляющей. Значительным недостатком является также то, что спутниковые системы имеют низкую помехоустойчивость. 
Целью работы является исследование эффективности методов машинного обучения для борьбы  с преднамеренными помехами, имитирующими сигнал, идущий от навигационного спутника (спуфинг). 

 <h2 style="text-align: center;">Технологический стек </h2> 

* Система: Ubuntu 22.04
* Инструменты контейнеризации: Docker
* Системы управления зависимостями: conda, pip
* Язык программирования: Python 3.10.12
* Машинное обучение: OneClassSVM (sklearn), IsolationForest (sklearn)
* Работа с данными:  NumPy, Pandas, StandardScaler, GridSearchCV
* Логирование экспериментов: ML_Flow

<h2 style="text-align: center;">Запуск в Docker </h2> 

 ```python
  docker-compose up 
```
<h2 style="text-align: center;">Запуск ML Flow </h2> 

Написать в терминале 
 ```python
conda activate your_env

mlflow server --host 127.0.0.1 --port 5000
 ```

 Инициализация ML_flow
```python
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('Spoof_test1')
```

Функция сохранения графиков в ML flow

```python
def log_current_figure_to_mlflow(fig, name="plot1"):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name)
        mlflow.log_artifact(tmpfile.name, artifact_path="plots")
        tmpfile_path = tmpfile.name
    os.remove(tmpfile_path)
```

 <h2 style="text-align: center;">Постановка задачи </h2> 

 Выявить воздействие спуфинга предлагается путем сопоставления данных ИНС и ГНСС. Считаем, что изначально ИНС находится в автономном режиме, т.е. не имеет возможности скорректироваться по искаженному спуфингом сигналу ГНСС. Коррекция ИНС производится по прошлым измерениям ГНСС на интервале, для которого диагностика подтвердила отсутствие спуфинга. 
Каждый вектор выходных параметров ИНС s(t)  и выходных параметров ГНСС  r(t)  представляет собой многомерные последовательности изменения широты fi(t)  и долготы lamda(t)  во времени:
<div id ="header" align="center">
 <img src="readme_content/form1.jpg" width ="300"/>
 </div>	
<div id ="header" align="center">
 <img src="readme_content/exp1.jpg" width ="300"/>
 </div>	
 <div id ="header" align="center">
 <img src="readme_content/exp2.jpg" width ="300"/>
 </div>	
<div id ="header" align="center">
 <img src="readme_content/exp3.jpg" width ="300"/>
 </div>	

Здесь <img src="readme_content/exp6.jpg" alt="alt text" width="80" height="40" /> -  погрешности ухода ИНС и погрешности ГНСС, вызванные несинхронностью часов, затуханием в атмосфере, собственными шумами приемника и переотражениями и т.д.
При воздействии спуфинга компоненты r(t) примут вид 
<div id ="header" align="center">
 <img src="readme_contentexp5.jpg" width ="300"/>
 </div>	

 где <img src="readme_content/exp6.jpg" alt="alt text" width="80" height="40" />  - погрешность, вызванная воздействием спуфинга. 

 Таким образом, на основе анализа невязки <img src="readme_content/exp7.jpg" alt="alt text" width="100" height="40" /> необходимо определить время начала и время окончания действия спуфинга, характеризующегося присутствием в сигнале ГНСС компоненты <img src="readme_content/exp6.jpg" alt="alt text" width="80" height="40" />
 
<h2 style="text-align: center;">Алгоритмы решения задачи </h2>

Использованы 2 алгоритма классификации спуфинга: OneClass SVM и Isolation Forest со следующими параметрами:

 ```python
 models = {
    "one_class_svm": OneClassSVM(nu=0.001, kernel = 'rbf', gamma = 0.0001),
    "isolation_forest": IsolationForest(n_estimators = 50, max_samples = 0.7, contamination = 0.01)
```
<h2 style="text-align: center;">Результаты </h2>

В имитаторе можно сгенерировать 3 вида спуфинга: синусоидальный, линейный и спиралевидный. Далее графики приведены для синусоидального спуфинга

График траекторий ИНС и ГНСС  с появлением синусоидальной ошибкой спустя 60 секунд с длительностью 60 секунд

<div id ="header" align="center">
 <img src="readme_content/fig1.jpg" width ="300"/>
 </div>	

 Графики невязок и приращений невязок ГНСС и ИНС с истинными классами и классами One-Class SVM

 <img src="readme_content/fig2.jpg" alt="alt text" width="300" height="300" /> <img src="readme_content/fig3.jpg" alt="alt text" width="300" height="300" />
  <img src="readme_content/fig4.jpg" alt="alt text" width="300" height="300" /> <img src="readme_content/fig5.jpg" alt="alt text" width="300" height="300" />

  Графики изменения невязок ГНСС и ИНС с истинными классами и классами One-Class SVM

   <img src="readme_content/fig6.jpg" alt="alt text" width="1000" height="300" /> 

   Графики значения функции прогноза One-Class SVM от времени

   <img src="readme_content/fig7.jpg" alt="alt text" width="1000" height="300" /> 

   Графики невязок и приращений невязок ГНСС и ИНС с истинными классами и классами Isolation Forest

 <img src="readme_content/fig8.jpg" alt="alt text" width="300" height="300" /> <img src="readme_content/fig9.jpg" alt="alt text" width="300" height="300" />
  <img src="readme_content/fig10.jpg" alt="alt text" width="300" height="300" /> <img src="readme_content/fig11.jpg" alt="alt text" width="300" height="300" />

  Графики изменения невязок ГНСС и ИНС с истинными классами и классами Isolation Forest

   <img src="readme_content/fig12.jpg" alt="alt text" width="1000" height="300" /> 

   Графики значения функции прогноза Isolation Forest от времени

   <img src="readme_content/fig13.jpg" alt="alt text" width="1000" height="300" /> 