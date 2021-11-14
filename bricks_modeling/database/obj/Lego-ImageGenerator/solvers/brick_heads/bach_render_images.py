import glob
import os

ldview_exe = r"/Applications/LDView.app/Contents/MacOS/LDView"


def render_ldrs(folder_path):

    for filepath in sorted(
        glob.glob(os.path.join(folder_path, "*.ldr")), key=os.path.getmtime
    ):
        print("rendering ldr file:", filepath)
        filename = os.path.basename(filepath)
        # os.system(f"{ldview_exe} {filepath} -SaveSnapshot={filepath}_top.png -DefaultLatLong=0,0 -AutoCrop=0")
        # os.system(f"{ldview_exe} {filepath} -SaveSnapshot={filepath}_front.png -DefaultLatLong=90,0")
        # os.system(f"{ldview_exe} {filepath} -SaveSnapshot={filepath}_left.png -DefaultLatLong=0,90")
        os.system(
            f"{ldview_exe} {filepath} -SaveSnapshot={filepath}.png -DefaultLatLong=40,40 -SaveAlpha=1 -ShowAxes=0 -FOV=12.0 -EdgeThickness=3.0"
        )


if __name__ == "__main__":
    folder_path = r"/Users/apple/Dropbox/deecamp/parts_library/hair/hair_images/testing"
    render_ldrs(folder_path)
