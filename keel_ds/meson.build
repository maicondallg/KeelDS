py = import('python').find_installation(pure: true)

py.install_sources(
  [
    '__init__.py',
    'load.py',
  ],
  subdir: 'keel_ds'
)

# Install data directories
subdir('data')
