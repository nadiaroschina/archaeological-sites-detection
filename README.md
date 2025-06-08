# Распознавание археологических объектов на спутниковых снимках Египта

На территории современного Египта и Судана сохранилось множество археологических объектов — укреплений, поселений и рабочих построек, относящихся к античному периоду. До сих пор поиск таких объектов велся экспертами-археологами вручную по спутниковым снимкам. Такой подход требует больших временных затрат и затрудняет исследование обширных территорий.

Задача данной работы — разработать инструмент автоматической фильтрации спутниковых снимков пустынной местности, позволяющий археологам сосредоточить внимание на наиболее перспективных участках. Сложность задачи обусловлена несколькими факторами:
- слабовыраженными визуальными признаками объектов на изображениях;
- низким качеством и контрастностью спутниковых данных;
- сильным дисбалансом между положительными (объекты) и отрицательными (фон) примерами;
- отсутствием размеченных данных в достаточном объеме.

Для решения задачи рассматриваются подходы из области детекции аномалий. Предполагается, что изображения с археологическими объектами можно трактовать как аномалии относительно «нормальной» пустынной местности. Основная цель — построить модель, оценивающую вероятность наличия археологического объекта на снимке, и использовать её в качестве фильтра, отбрасывающего заведомо неинформативные изображения.

<div style="display: flex; justify-content: center; gap: 10px;">

  <div style="width: 50%; text-align: center;">
    <img src="./Deir_El-Atrash_10_2022.jpg" alt="Deir_El-Atrash" style="width: 100%;">
    <p><em>Deir El-Atrash. Крупная крепость</em></p>
  </div>

  <div style="width: 50%; text-align: center;">
    <img src="./Umm_Huyut_11_2022.jpg" alt="Umm_Huyut" style="width: 100%;">
    <p><em>Umm Huyut. Постройки возле карьера</em></p>
  </div>

</div>

## Данные

Данные для основного датасета собирались из сервисов Yandex Tiles API, Google Tiles API и QGIS.
Все данные и скрипты для подготовки датасетов тайликов лежат в директории [data_retrieval](https://github.com/nadiaroschina/archeological-sites-detection/tree/main/data_retrieval), а скрипты для работы с s3 хранилищем и yandex datasphere datasets - в директории [datasphere_data_manager](https://github.com/nadiaroschina/archeological-sites-detection/tree/main/datasphere_data_manager)

Координаты известных археологических объектов хранятся в следующих файла

| Описание                                                                                   | Название файла                                                |
|--------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| 200 археологических объектов                                                              | [Eastern_desert_archaeological_structures.csv](https://github.com/nadiaroschina/archeological-sites-detection/tree/main/data_retrieval/coordinates_data/Eastern_desert_archaeological_structures.csv)               |
| 150 объектов, хорошо видимых на снимках Google Satellite, без современных построек         | [Eastern_desert_archaeological_structures_NEW.csv](https://github.com/nadiaroschina/archeological-sites-detection/tree/main/data_retrieval/coordinates_data/Eastern_desert_archaeological_structures_NEW.csv)            |
| 50 объектов, по которым есть подготовленный археологами датасет спутниковых снимков       | [Satellite_images_sites_table.csv](https://github.com/nadiaroschina/archeological-sites-detection/blob/main/data_retrieval/coordinates_data/Satellite_images_sites_table.csv)                           |



## Пайплайны обучения моделей

<table>
  <thead>
    <tr>
      <th>Подход</th>
      <th>Описание</th>
      <th>Группа моделей</th>
      <th>Директория с ноутбуками для обучения</th>
      <th>Отдельная директория с результатами</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Геометрический</b></td>
      <td>Выделение границ объектов с помощью оператора Кэнни и приближение многоугольниками</td>
      <td>Canny</td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/canny_edge_detection">canny_edge_detection</a></td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/canny_edge_detection/edge_detection_results">edge_detection_results</a></td>
    </tr>
    <tr>
      <td><b>Классификационный</b></td>
      <td>Добучение предобученного ConvNeXt на синтетических примерах аномалий</td>
      <td>ConvNeXt</td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/artificial_data_generation">artificial_data_generation</a><br><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/convnext_on_artificial_data">convnext_on_artificial_data</a></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Реконструкционный</b></td>
      <td rowspan="4">Обучение автокодировщика для восстановление изображений и метод детекции аномалий по исходному и восстановленному изображениям</td>
      <td>VAE и RVAE</td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/vae_training_yandextiles">vae_training_yandextiles</a><br><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/vae_training_googletiles">vae_training_googletiles</a><br></td>
      <td></td>
    </tr>
    <tr>
      <td>fAnoGAN</td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/fanogan_training">fanogan_training</a></td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/fanogan_training_results">fanogan_training_results</a></td>
    </tr>
    <tr>
      <td>VQ-VAE</td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/vq_vae_training_googletiles">vq_vae_training_googletiles</a><br><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/vq_vae_training_sasgis">vq_vae_training_sasgis</td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/vq_vae_training_googletiles_results">vq_vae_training_googletiles_results</a><br><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/vq_vae_training_sasgis_results">vq_vae_training_sasgis_results</a></td>
    </tr>
    <tr>
      <td>Детекция аномалий</td>
      <td><a href="https://github.com/nadiaroschina/archeological-sites-detection/tree/main/anomaly_detection_models">anomaly_detection_models</a></td>
      <td></td>
    </tr>
  </tbody>
</table>
