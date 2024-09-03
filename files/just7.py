import pathlib
import sys
import glob
import os
import numpy as np
import cv2
import tensorflow as tf
import time
import random


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
import carla

# Initialize global variables
actor_list = []
agent = None
running = True
detection_model = None
category_index = None

# def get_random_spawn_point(world):
#     """Gets a random spawn point from the world."""
#     spawn_points = world.get_map().get_spawn_points()
#     random_index = random.randint(0, len(spawn_points) - 1)
#     return spawn_points[random_index]

def load_model(model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        print(f'Error: Model directory {model_dir} does not exist.')
        return None

    try:
        model = tf.saved_model.load(str(model_dir))
        print('Model loaded successfully.')
        return model
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        return None

model_dir = r"C:\Carla\CARLA_0.9.8\WindowsNoEditor\PythonAPI\examples\saved_model"
detection_model = load_model(model_dir)
if detection_model is None:
    sys.exit('Model could not be loaded. Exiting.')

def create_category_index_from_labelmap(label_map_file_path):
    if not os.path.exists(label_map_file_path):
        raise FileNotFoundError(f'Label map file not found: {label_map_file_path}')

    label_map = {}
    with open(label_map_file_path, 'r') as f:
        label = None
        for line in f:
            if 'id:' in line:
                id = int(line.split('id:')[1].strip())
            if 'name:' in line:
                name = line.split('name:')[1].strip().replace("'", "")
                label_map[id] = {'id': id, 'name': name}
    return label_map

label_map_file_path = r"C:\Carla\CARLA_0.9.8\WindowsNoEditor\PythonAPI\examples\driving_label_map.pbtxt"
category_index = create_category_index_from_labelmap(label_map_file_path)
print('Category index created:', category_index)

def show_inference(model, frame):
    print('Running inference')
    if frame is None:
        print("Error: Frame is None in show_inference.")
        return None

    output_dict = run_inference_for_single_image(model, frame)
    frame = visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=2)

    return frame

def run_inference_for_single_image(model, image):
    print('Start inference')
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    print('End inference')
    return output_dict

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    def reframe_box_masks_to_image_masks_default(box_masks, boxes, image_height, image_width):
        num_boxes = boxes.shape[0]
        image_masks = np.zeros([num_boxes, image_height, image_width], dtype=np.float32)
        for i in range(num_boxes):
            box = boxes[i]
            box_mask = cv2.resize(box_masks[i], (int(box[3] - box[1]), int(box[2] - box[0])))
            image_masks[i, int(box[0]):int(box[2]), int(box[1]):int(box[3])] = box_mask
        return image_masks

    box_masks = tf.image.resize(box_masks, [image_height, image_width])
    return reframe_box_masks_to_image_masks_default(box_masks, boxes, image_height, image_width)

def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,
                                              instance_masks=None, use_normalized_coordinates=False,
                                              line_thickness=2):
    if image is None:
        print("Error: Image is None in visualize_boxes_and_labels_on_image_array.")
        return None

    for i in range(boxes.shape[0]):
        if scores[i] > 0.5:
            box = tuple(boxes[i].tolist())
            class_id = classes[i]
            print(f'Class ID: {class_id}')  # Debug print to check class IDs
            if class_id not in category_index:
                print(f'Class ID {class_id} not found in category index.')
                continue
            class_name = category_index[class_id]['name']
            display_str = f'{class_name}: {int(100 * scores[i])}%'
            image = draw_bounding_box_on_image_array(image, box[0], box[1], box[2], box[3], display_str)
    return image

def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, display_str,
                                     color='red', thickness=4):
    if image is None:
        print("Error: Image is None in draw_bounding_box_on_image_array.")
        return None

    # Convert color name to BGR tuple
    color_map = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
    if color in color_map:
        color = color_map[color]
    else:
        raise ValueError(f"Color '{color}' not recognized. Use 'red', 'green', or 'blue'.")

    (im_width, im_height) = image.shape[1], image.shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, thickness)
    cv2.putText(image, display_str, (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

# def process_image(image, idx):
#     global running
#     image.convert(carla.ColorConverter.Raw)
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]  # Extract RGB channels
#     processed_image = show_inference(detection_model, array)
#     cv2.imshow(f'Camera Feed {idx}', processed_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         running = False

def processImage(image):
	image = np.array(image.raw_data)
	img = image.reshape((600,800,4))
	img = img[:,:,:3]

	img = show_inference(detection_model , img)
	cv2.imshow("img",img)
	 
	# cv2.imshow("gray img",gray_img) 
	cv2.waitKey(1)
	return img

def cleanup():
    global actor_list
    for actor in actor_list:
        actor.destroy()
    print("All actors cleaned up!")
 
# Main function to initialize and run the simulation
def main():
    global actor_list, detection_model, category_index, running, agent, vehicle

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    # To start a basic agent
    # agent = BasicAgent(vehicle)


    # Get vehicle blueprint and spawn point
    vehicle_bp = blueprint_library.filter('cybertruck')[0]
    # spawn_point = get_random_spawn_point(world)
    
    # Define start and destination locations manually
    start_location = carla.Transform(carla.Location(x=130, y=195, z=40))
    destination_location = carla.Location(x=200, y=250, z=40)

    # Spawn the vehicle at the start location
    vehicle = world.spawn_actor(vehicle_bp, start_location)
    agent = BasicAgent(vehicle)

    # Set the destination for the agent
    # destination = random.choice(spawn_points).location
    # agent.set_destination(destination_location)
    
    # Convert destination_location to a tuple of coordinates (x, y, z)
    destination_coords = (destination_location.x, destination_location.y, destination_location.z)
    
    # Set the destination for the agent
    agent.set_destination(destination_coords)

    actor_list.append(vehicle)

    # Set autopilot
    vehicle.set_autopilot(True)

    # Camera attributes
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')

    # Camera location and rotation
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(lambda image: processImage(image, 0))
    actor_list.append(camera)

    try:
        while running:
            time.sleep(0.05)  # Sleep to reduce CPU usage
    except KeyboardInterrupt:
        running = False
    finally:
        cleanup()

if __name__ == '__main__':
    main()
