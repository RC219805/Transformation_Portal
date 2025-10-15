# Terminal commands to reproduce the Montecito batch workflow. Any Python snippets
# have been moved into ``tools/montecito_manifest.py`` so this history remains
# shell-friendly.
npm install -g npm@11.6.0
npm install -g @openai/codex
python3 luxury_tiff_batch_processor.py "/Users/rc/Montecito-batch" \
  "/Users/rc/Montecito-batch-out" --preset signature --suffix _lux --overwrite
python3 luxury_tiff_batch_processor.py "/Users/rc/Montecito-batch" \
  "/Users/rc/Montecito-batch-out" --preset signature --suffix _lux --overwrite
python3 luxury_tiff_batch_processor.py "/Users/rc/Montecito-batch" \
  "/Users/rc/Montecito-batch-out" --preset signature --suffix _lux --overwrite
pip install numpy
pip install numpy pillow tifffile
python3 -m pip install --no-index --find-links "$HOME/Downloads" pytest
python3 -m pip install --no-index --find-links /Users/rc/Downloads pytest
pip install pytest
pip3 install --no-index --find-links … numpy
python3 --version
pip3 --version
mkdir my_pytest_project
cd my_pytest_project
python3 -m venv venv_pytest
source venv_pytest/bin/activate
pytest --version
deactivate
python3 -m pip install --no-index --find-links "$HOME/Downloads" pytest
brew install pytest
brew install pipx
pipx install --no-index --find-links /Users/rc/Downloads pytest
hash -r
which pytest
pytest --version
zip -r -s 90m montecito_out.zip .
python3 tools/montecito_manifest.py "/Users/rc/Montecito-batch-out" \
  "/Users/rc/Montecito-batch-out/manifest.csv"
brew install gh
echo 'eval "$(gh copilot alias -- bash)"' >> ~/.bashrc
brew install gitkraken-cli
