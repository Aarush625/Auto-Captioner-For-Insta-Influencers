from PIL import ImageFont
import os

font_dirs = [os.path.join(os.environ['WINDIR'], 'Fonts')]  # Windows only
fonts = [f for f in os.listdir(font_dirs[0]) if f.endswith(".ttf")]

for f in fonts[:10]:  # Print first 10 fonts
    print(f)
