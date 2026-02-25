conda create -n iven python=3.13
conda activate iven
pip install -r requirements.txt

pyinstaller iven.py --onefile --windowed --name "IVEN" --add-data "assets:assets" --icon logos/iven_mac.png