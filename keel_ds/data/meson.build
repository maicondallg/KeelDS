py = import('python').find_installation(pure: true)

# Install data files for balanced datasets
install_subdir(
  'balanced/processed',
  install_dir: py.get_install_dir() / 'keel_ds/data/balanced'
)

# Install data files for imbalanced datasets
install_subdir(
  'imbalanced/processed',
  install_dir: py.get_install_dir() / 'keel_ds/data/imbalanced'
)

# Install preset files
install_subdir(
  'presets',
  install_dir: py.get_install_dir() / 'keel_ds/data'
)
