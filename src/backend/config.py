#!/usr/bin/env python3
from os import path

base_directory = path.join(path.dirname(path.abspath(__file__)), '..', '..')
log_file_path = path.join(base_directory, 'logs', 'app.log')
models_path = path.join(base_directory, 'models')
datasets_path = path.join(base_directory, 'datasets')
template_directory_path = path.join(base_directory, 'src', 'frontend', 'templates')
static_assets_directory_path = path.join(base_directory, 'src', 'frontend', 'static')