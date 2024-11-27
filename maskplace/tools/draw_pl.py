
from place_db import PlaceDB

placedb = PlaceDB("adaptec1")
placedb.debug_str()


output_path = 'output_image.png'  # Replace with your desired output image path

placedb.save_fig(output_path)

print(f"Visualization saved as {output_path}")
