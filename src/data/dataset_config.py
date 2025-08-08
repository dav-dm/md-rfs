from util.config import load_config

cf = load_config()
_BASE_DATA_PATH = cf['base_data_path']

dataset_config = {
    'iot23': {
        'path': (
            f'{_BASE_DATA_PATH}/iot23/'
            'iot23.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'cic2018': {
        'path': (
            f'{_BASE_DATA_PATH}/cic2018/'
            'cic2018.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'insdn': {
        'path': (
            f'{_BASE_DATA_PATH}/in_sdn/'
            'in_sdn.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
}
