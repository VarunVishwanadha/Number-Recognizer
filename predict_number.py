import tensorflow as tf
import random
import os
import math


def predict_digit(input_file, number_of_digits):
    if os.path.exists(input_file) == False:
        print("File does not exist")
        exit(0)

    file_directory = os.getcwd()
    layer_nodes = [784, 800, 10]
    x = tf.placeholder(tf.float32, [None, 784])
    w = []
    b = []
    y_list = []

    for idx in range(len(layer_nodes)-1):
        w.append(tf.get_variable("w"+str(idx), shape=[layer_nodes[idx], layer_nodes[idx+1]]))
        b.append(tf.get_variable("b"+str(idx), shape=[layer_nodes[idx+1]]))
        if idx == 0:
            y_list.append(tf.nn.sigmoid(tf.matmul(x, w[idx]) + b[idx]))
        else:
            y_list.append(tf.nn.sigmoid(tf.matmul(y_list[idx-1], w[idx]) + b[idx]))

    y = y_list[len(y_list)-1]

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    try:
        saver.restore(sess, file_directory+'/model.ckpt')
    except:
        print("Please train the model first. Run train_model.py")
        exit(0)

    filename_queue = tf.train.string_input_producer([file_directory+'/' + input_file])

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    input_image = tf.image.decode_png(value, channels=1)
    resized_image = tf.image.resize_images(input_image, (28, 28))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image = resized_image.eval()

    coord.request_stop()
    coord.join(threads)

    reshaped_image = image.reshape(1, 784)
    for idx in range(len(reshaped_image[0])):
        reshaped_image[0][idx] = 255 - reshaped_image[0][idx]

    if number_of_digits > 1:
        clustered_images = get_clustered_images(reshaped_image[0], number_of_digits)
    else:
        clustered_images = [reshaped_image]

    predicted_number = 0
    for idx, value in enumerate(clustered_images):
        predicted_digit = sess.run(y, feed_dict={x: value})
        digit = tf.argmax(predicted_digit, 1).eval()[0]
        predicted_number += math.pow(10, number_of_digits-idx-1)*digit
    print("Input file  - ", input_file)
    print("Predicted Number  - ", int(predicted_number))


def get_clustered_images(image, number_of_clusters):
    data_points = get_data_points(image)
    centroids = get_initial_centroids(image, data_points, number_of_clusters)
    max_iterations = 30
    clustered_images = []
    for idx in range(max_iterations):
        for j in range(number_of_clusters):
            centroids[j]['points'] = []
        for index in data_points:
            nearest_centroid_index = get_nearest_centroid_index(index, centroids)
            centroids[nearest_centroid_index]['points'].append(index)
        new_centroids_obj = get_new_centroids(centroids, image)
        if new_centroids_obj['isUpdated']:
            centroids = new_centroids_obj['centroids']
        else:
            break

    centroids.sort(key=lambda x: x['index'] % 28)

    for i in range(number_of_clusters):
        centroids_x = [x%28 for x in centroids[i]['points']]
        min_index = min(centroids_x)
        max_index = max(centroids_x) + 1
        # print("Min, max points ", i, min_index, max_index)
        # print("Centroid x, y", i, centroids[i]['index'] % 28, math.floor(centroids[i]['index']/28))
        img = []
        for idx, value in enumerate(image):
            # Crop the image to ignore whitespace
            if idx % 28 in range(min_index, max_index):
                img.append(value)
        img = tf.convert_to_tensor(img).eval().reshape(28, max_index-min_index, 1)

        # Add padding to the image to center the digit
        padding_left = tf.zeros([28, 5, 1], tf.float32)
        padding_right = tf.zeros([28, 5, 1], tf.float32)
        tensor_img = tf.concat([padding_left, img, padding_right], 1)

        # Resize the image to 28*28 pixels
        resized_image = tf.image.resize_images(tensor_img, (28, 28))
        clustered_images.append(resized_image.eval().reshape(1, 784))
    return clustered_images


# Get all non-zero data points
def get_data_points(image):
    data_points = []
    for index, value in enumerate(image):
        if value != 0:
            data_points.append(index)
    return data_points


def get_initial_centroids(image, data_points, number_of_clusters):
    centroids = []
    for idx in range(number_of_clusters):
        centroids.append({'index': 0, 'points': []})
    
    weighted_sum_of_x = 0
    weighted_sum_of_y = 0
    sum_of_weights = 0
    min_of_x = 27
    max_of_x = 0

    for index in data_points:
        x_coordinate = index % 28
        weighted_sum_of_x += x_coordinate*image[index]
        weighted_sum_of_y += math.floor(index/28)*image[index]
        sum_of_weights += image[index]
        if x_coordinate < min_of_x:
            min_of_x = x_coordinate
        if x_coordinate > max_of_x:
            max_of_x = x_coordinate
            
    if sum_of_weights != 0:
        center_of_gravity_x = math.floor(weighted_sum_of_x/sum_of_weights)
        center_of_gravity_y = math.floor(weighted_sum_of_y/sum_of_weights)
        max_offset_x = min(center_of_gravity_x-min_of_x, max_of_x-center_of_gravity_x)
        for i in range(number_of_clusters):
            centroids[i]['index'] = center_of_gravity_x + max_offset_x*(2*i/(number_of_clusters-1) - 1) + 28*center_of_gravity_y
            # centroids[i]['index'] = random.randint(0,784)

    return centroids


def get_nearest_centroid_index(point, centroids):
    centroid_distance = []
    for centroid in centroids:
        distance = abs(centroid['index'] % 28 - point % 28)
        # distance = math.sqrt(math.pow((centroid['index'] % 28 - point % 28), 2) + math.pow((math.floor(centroid['index']/28) - math.floor(point/28)), 2))
        centroid_distance.append(distance)
    min_distance_index = centroid_distance.index(min(centroid_distance))
    return min_distance_index


def get_new_centroids(centroids, image):
    new_centroids_obj = dict(centroids=centroids, isUpdated=False)
    for centroid in new_centroids_obj['centroids']:
        weighted_sum_of_x = 0
        weighted_sum_of_y = 0
        sum_of_weights = 0
        for point in centroid['points']:
            weighted_sum_of_x += (point%28)*image[point]
            weighted_sum_of_y += math.floor(point/28)*image[point]
            sum_of_weights += image[point]
        if sum_of_weights != 0:
            centroid_x = math.floor(weighted_sum_of_x/sum_of_weights)
            # centroid_y = math.floor(weighted_sum_of_y/sum_of_weights)
            centroid_y = 14
            if centroid['index'] != centroid_x + 28*centroid_y:
                new_centroids_obj['isUpdated'] = True
            centroid['index'] = centroid_x + 28*centroid_y

    return new_centroids_obj


predict_digit('Sample Images/192.png', 3)
