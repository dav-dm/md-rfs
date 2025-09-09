# Hyperparameter Configurations for Training Approaches

This folder contains `.json` files with the hyperparameters used to train the various approaches presented in the [paper]().
It also includes the file `dataset_sample.parquet`, which serves as an example of the required dataset format for use with this framework.

> **Note:** For few-shot learning experiments, simply add the option `--k` when running the commands.

---

## Example Commands to Run Different Approaches

### ML Approaches

- **Random Forest**  
```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach random_forest --seed 0 --is-flat
````

* **XGBoost**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach xgb --seed 0 --is-flat
```

### DL Approaches

* **2D-CNN**

```bash
python main.py --src-dataset cic2018 --trg-dataset iot23 --approach baseline --max-epochs 200 --n-task 1 --seed 0 --log-dir ADD_A_PATH
```

* **Fine Tuning**

```bash
python main.py --src-dataset cic2018 --trg-dataset iot23 --approach baseline --adaptation-strat finetuning --max-epochs 200 --adapt-epochs 200 --n-task 2 --seed 0 --log-dir ADD_A_PATH
```

* **Freezing**

```bash
python main.py --src-dataset cic2018 --trg-dataset iot23 --approach baseline --adaptation-strat freezing --max-epochs 200 --adapt-epochs 200 --n-task 2 --seed 0 --log-dir ADD_A_PATH
```

* **MD-RFS**

  After training the teacher model and the domain discriminator autoencoder, run:

```bash
python main.py --src-dataset cic2018 --trg-dataset iot23 --approach md_rfs --seed 0 --max-epochs 200 --n-tasks 2 --teacher-path ADD_A_PATH --discr-path ADD_A_PATH
```

---

Replace `ADD_A_PATH` with the actual paths to your log directories or saved models.
