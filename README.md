# Segmentacja Skał Księżycowych (Moon Rock Segmentation)
Projekt zakłada semantyczną segmentację skał na powierzchni księżyca. Celem jest detekcja i klasyfikacja skał małych, dużych oraz nieba na fotorealistycznych renderach.
Projekt wykorzystuje bibliotekę `segmentation-models-pytorch` do implementacji architektury Linknet opartej na enkoderze ResNet. 

## Zbiór Danych (Dataset)


Projekt wykorzystuje zbiór "Artificial Lunar Landscape Dataset" [[DOI](https://doi.org/10.34740/kaggle/dsv/13263000)].

- **Wejście:** Obrazy RGB (Rendery)
- **Wyjście:** Maski segmentacji
- **Klasy:**
    1. **Sky (0):** Tło, przestrzeń kosmiczna, horyzont.
    2. **Small Rocks (1):** Mniejsze odłamki skalne i regolit.
    3. **Large Rocks (2):** Głównie duże głazy i formacje skalne.

### Czyszczenie Danych
Zbiór danych zawiera znane błędy (artefakty renderowania, niedopasowanie masek). W preprocessingu usuwane są próbki zidentyfikowane jako wadliwe.

## Architektura Modelu

W domyślnej konfiguracji projekt wykorzystuje:

- **Architektura:** Linknet
- **Enkoder:** ResNet34 (z wagami pretrenowanymi na ImageNet)
- **Funkcja aktywacji:** Softmax2d 
- **Funkcja straty:** DiceLoss + CrossEntropyLoss (hybrydowe podejście)


Wybór architektury został oparty o porównanie modeli segmentacyjnych przedstawione w materiale: ["Semantic Segmentation Models: Comparison" (YouTube)](https://www.youtube.com/watch?v=pw6Jz4lX2Kc).

### Instalacja zależności

```bash
pip install -r requirements.txt
```

Główne biblioteki:
- `torch` 
- `segmentation-models-pytorch` 
- `albumentations` 
- `wandb` 
- `opencv-python`, `pandas`, `numpy`

## Konfiguracja i Użycie

Wszelkie hiperparametry treningu znajdują się w pliku `configs/base_config.yaml`.

### Struktura pliku konfiguracyjnego
```yaml
model:
  architecture: "Linknet"
  encoder: "resnet34"      
  weights: "imagenet"
  in_channels: 3
  num_classes: 4           

training:
  batch_size: 16
  epochs: 50
  lr: 0.0001
  optimizer: "AdamW"
  loss: "dice_ce"
  num_workers: 4
  device: "auto"       
```

### Trening
Aby rozpocząć proces uczenia modelu, należy uruchomić skrypt treningowy.

```bash
python src/train.py 
```
Skrypt automatycznie inicjalizuje logowanie do Weights & Biases (WandB), jeśli jest skonfigurowane.

### Ewaluacja i Inferencja
Generowanie masek dla zbioru testowego odbywa się za pomocą skryptu `inference.py`.

```bash
python src/inference.py 
```


## Struktura Repozytorium

- **[`configs/base_config.yaml`](configs/base_config.yaml)**: Plik konfiguracyjny z hiperparametrami i ustawieniami modelu.
- **[`data/`](data/)**: Surowe dane, listy anomalii oraz pliki manifestu CSV.
  - `archive/`: Zbiór obrazów i masek, pliki pomocnicze (np. bounding_boxes.csv, listy anomalii).
- **[`src/`](src/)**: Kod źródłowy projektu.
  - [`dataset.py`](src/dataset.py): Klasa `MoonDataset` i logika ładowania danych.
  - [`train.py`](src/train.py): Główna pętla treningowa.
  - [`inference.py`](src/inference.py): Skrypt do generowania predykcji masek.
  - [`preprocess.py`](src/preprocess.py): Skrypty do wstępnego przetwarzania danych.
  - [`verify.py`](src/verify.py): Skrypty do weryfikacji poprawności danych.

