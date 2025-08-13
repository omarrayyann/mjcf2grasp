import os
import platform
import argparse
import base64
import io
from flask import Flask, render_template_string, request, jsonify
import mujoco
from PIL import Image


def get_objects(directory):
    objects = []

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return objects

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml") and "old" not in file:
                full_path = os.path.join(root, file)
                filename_without_ext = os.path.splitext(file)[0]
                objects.append((filename_without_ext, full_path))

    return objects


if platform.system() != "Darwin":
    os.environ["MUJOCO_GL"] = "osmesa"

parser = argparse.ArgumentParser(
    description="XML Objects Viewer - View all XML files in a directory recursively"
)
parser.add_argument("directory", help="Directory to search for XML files recursively")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

debug_mode = args.debug
app = Flask(__name__)

if not os.path.exists(args.directory):
    print(f"Error: Directory '{args.directory}' does not exist")
    exit(1)

print(f"Scanning directory '{args.directory}' for XML files...")
objects = sorted(get_objects(args.directory), key=lambda x: x[0])
print(f"Found {len(objects)} XML files")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>XML Objects Viewer</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        input { padding: 5px; width: 300px; margin-bottom: 10px; }
        .item { cursor: pointer; color: blue; text-decoration: underline; margin: 5px 0; }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background-color: rgba(0,0,0,0.8);
        }
        .modal-content {
            position: relative;
            background: #fff;
            border-radius: 8px;
            width: 90vw;
            height: 90vh;
            margin: 5vh auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 40px 20px 20px 20px;
            box-sizing: border-box;
        }
        .close {
            position: absolute;
            top: 10px;
            right: 20px;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            z-index: 10;
        }
        #object-image {
            max-width: 100%;
            max-height: 100%;
            border: 1px solid #ccc;
            object-fit: contain;
        }
        #loading {
            display: none;
            font-size: 16px;
            color: #555;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>XML Objects Viewer</h1>
    <p>Viewing XML files from: <strong>{{ directory }}</strong></p>
    <p>Found <strong>{{ total_count }}</strong> XML files</p>
    <input type="text" id="search" placeholder="Search XML files..." onkeyup="filterList()">
    <div id="list">
        {% for name in object_names %}
            <div class="item" onclick="loadImage('{{name}}')">{{name}}</div>
        {% endfor %}
    </div>

    <div id="imageModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="loading">Loading image...</div>
            <div id="camera-controls" style="display:none; position:absolute; top:50px; left:20px; z-index:20;">
                <button onclick="zoomIn()" style="margin:5px; padding:10px; font-size:14px; cursor:pointer;">Zoom In</button>
                <button onclick="zoomOut()" style="margin:5px; padding:10px; font-size:14px; cursor:pointer;">Zoom Out</button>
            </div>
            <img id="object-image" src="" style="display:none">
        </div>
    </div>

    <script>
        let currentObjectName = null;
        let currentZoomLevel = 1.0;

        function loadImage(name) {
            currentObjectName = name;
            currentZoomLevel = 1.0;
            const img = document.getElementById("object-image");
            const loading = document.getElementById("loading");
            const controls = document.getElementById("camera-controls");
            
            img.style.display = "none";
            loading.style.display = "block";
            controls.style.display = "none";
            document.getElementById("imageModal").style.display = "block";

            renderWithZoom(name, currentZoomLevel);
        }

        function renderWithZoom(name, zoomLevel) {
            const img = document.getElementById("object-image");
            const loading = document.getElementById("loading");
            const controls = document.getElementById("camera-controls");
            
            loading.style.display = "block";
            img.style.display = "none";

            fetch('/render', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({object_name: name, zoom_level: zoomLevel})
            })
            .then(res => res.json())
            .then(data => {
                if (data.image_url) {
                    img.onload = () => {
                        loading.style.display = "none";
                        img.style.display = "block";
                        controls.style.display = "block";
                    };
                    img.src = data.image_url;
                }
            });
        }

        function zoomIn() {
            if (currentObjectName && currentZoomLevel > 0.3) {
                currentZoomLevel *= 0.7;
                renderWithZoom(currentObjectName, currentZoomLevel);
            }
        }

        function zoomOut() {
            if (currentObjectName && currentZoomLevel < 3.0) {
                currentZoomLevel *= 1.4;
                renderWithZoom(currentObjectName, currentZoomLevel);
            }
        }

        function closeModal() {
            document.getElementById("imageModal").style.display = "none";
            document.getElementById("camera-controls").style.display = "none";
            currentObjectName = null;
        }

        function filterList() {
            const q = document.getElementById("search").value.toLowerCase();
            document.querySelectorAll(".item").forEach(el => {
                el.style.display = el.textContent.toLowerCase().includes(q) ? "block" : "none";
            });
        }
    </script>
</body>
</html>
"""


def render_object_image(object_name, object_path, zoom_level=1.0):
    base_distance = 0.4
    camera_distance = base_distance * zoom_level
    camera_height = 1.0 * zoom_level

    scene_xml = f"""
<mujoco>
    <visual>
        <global offwidth="960" offheight="720" />
    </visual>
    <asset>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
            rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
            width="300" height="300"/>
        <material name="groundplane" texture="groundplane"
            texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>
    <option cone="elliptic" noslip_iterations="2" gravity="0 0 0" impratio="10">
        <flag multiccd="enable" />
    </option>
    <include file="{object_path}" />
    <worldbody>
        <light pos="0 0 1" dir="0 0 -1" directional="true"/>
        <light pos="0 1 0" dir="0 -1 0" directional="true"/>
         <light pos="0 -1 0" dir="0 1 0" directional="true"/>
        <light pos="1 0 0" dir="-1 0 0" directional="true"/>
        <light pos="-1 0 0" dir="1 0 0" directional="true"/>
        <camera name="main" pos="0.0 {camera_distance} {camera_height}" euler="-0.3 0 0" />
    </worldbody>
</mujoco>
    """

    try:
        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, width=960, height=720)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, "main")
        rgb = renderer.render()

        img = Image.fromarray(rgb)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_data}"
    except Exception as e:
        print(f"Error rendering {object_name}: {e}")
        return None


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        object_names=[name for name, _ in objects],
        directory=args.directory,
        total_count=len(objects),
    )


@app.route("/render", methods=["POST"])
def render_image():
    obj_name = request.json.get("object_name")
    zoom_level = request.json.get("zoom_level", 1.0)

    for name, path in objects:
        if name == obj_name:
            url = render_object_image(name, path, zoom_level)
            if url:
                return jsonify({"image_url": url})
            else:
                return jsonify({"error": f"Failed to render {obj_name}"}), 500
    return jsonify({"error": "Object not found"}), 404


if __name__ == "__main__":
    if len(objects) == 0:
        print(f"No XML files found in directory '{args.directory}'")
        print("Make sure the directory contains .xml files")
        exit(1)

    print(f"Starting server with {len(objects)} XML files...")
    print(f"Directory: {args.directory}")
    print("Visit http://localhost:5000 to view the XML objects")
    app.run(debug=debug_mode)
