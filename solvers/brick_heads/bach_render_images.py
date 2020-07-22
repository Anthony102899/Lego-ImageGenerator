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
            f"{ldview_exe} {filepath} -SaveSnapshot={filepath}_normal.png -DefaultLatLong=30,30"
        )


if __name__ == "__main__":
    folder_path = r"/Users/apple/workspace/lego-technic-solver/debug/2020-07-21_21-57-35_brick_heads"
    render_ldrs(folder_path)
