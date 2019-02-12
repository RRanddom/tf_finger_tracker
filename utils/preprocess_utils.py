import tensorflow as tf

        # input_image, points = random_left_right_flip(input_image, bbox, points, width, height)


dest_width = 488
dest_height = 488

def random_left_right_flip(image_tensor, bbox_tensor, points_tensor, image_width, image_height, prob=.3):
    """ Randomly left_right flips the image_tensor ,along with the points

    Args:
        image_tensor, tensor of image, shape = [height, width, 3]
        bbox_tensor,  tensor of bbox, shape=[2,2]
        points_tensor, tensor of points. shape = [3, 2],
        image_width, float or tensor type.
        image_height, float or tensor type.
        prob, float point number ~[0,1]. 
    
    Returns:
        image_tensor_output: same format as image_tensor
        points_tensor_output: same format as points_tensor
    """
    random_value = tf.random_uniform([])
    is_flipped = tf.less_equal(random_value, prob)
    
    def flip():
        image_tensor_reversed = tf.reverse_v2(image_tensor, [1])

        bbox_x_coords = image_width - bbox_tensor[:,0]
        bbox_y_coords = bbox_tensor[:,1]

        pts_x_coords = image_width - points_tensor[:,0]
        pts_y_coords = points_tensor[:,1]

        bbox_ret = tf.stack([bbox_x_coords, bbox_y_coords], axis=1)
        pts_ret = tf.stack([pts_x_coords, pts_y_coords], axis=1)

        return image_tensor_reversed, bbox_ret, pts_ret

    return tf.cond(is_flipped, flip, lambda: (image_tensor, bbox_tensor, points_tensor))

def random_crop(image_tensor, bbox_tensor, points_tensor):#image_width, image_height
    """ Randomly crop the 
    """
    x_min = tf.reduce_min(bbox_tensor[:,0])
    x_max = tf.reduce_max(bbox_tensor[:,0])

    y_min = tf.reduce_min(bbox_tensor[:,1])
    y_max = tf.reduce_max(bbox_tensor[:,1])

    # x belongs to [x_max-dest_width, x_min]
    # y belongs to [y_max-dest_height, y_min]

    rand_x = tf.cast((tf.random_uniform([])*(x_max-dest_width-x_min) + x_max-dest_width), tf.int)
    rand_y = tf.cast(tf.random_uniform([])*(y_max-dest_height-y_min) + y_max-dest_height, tf.int)

    image_cropped = image_tensor[rand_y:rand_y+dest_height, rand_x:rand_x+dest_width, :]

    bbox_x_coords = bbox_tensor[:,0] - rand_x
    bbox_y_coords = bbox_tensor[:,1] - rand_y

    pts_x_coords = points_tensor[:,0] - rand_x
    pts_y_coords = points_tensor[:,1] - rand_y

    return image_cropped, tf.stack([bbox_x_coords, bbox_y_coords], axis=1), tf.stack([pts_x_coords, pts_y_coords], axis=1)


# def random_up_down_flip(image_tensor, points_tensor, image_width, image_height, prob=.2):

#     """ Randomly up_down flips the image_tensor ,along with the points

#     Args:
#         image_tensor, tensor of image, shape = [height, width, 3]
#         points_tensor, tensor of points. shape = [4, 2],
#         image_width, float or tensor type.
#         image_height, float or tensor type.
#         prob, float point number ~[0,1]. 
    
#     Returns:
#         image_tensor_output: same format as image_tensor
#         points_tensor_output: same format as points_tensor
#     """
#     random_value = tf.random_uniform([])
#     is_flipped = tf.less_equal(random_value, prob)
    
#     def flip():
#         image_tensor_reversed = tf.reverse_v2(image_tensor, [0])

#         x_coords = points_tensor[:,0]
#         y_coords = image_height - points_tensor[:,1]
#         tmp = tf.stack([x_coords, y_coords], axis=1)
#         p1 = tmp[0,:]
#         p2 = tmp[1,:]
#         p3 = tmp[2,:]
#         p4 = tmp[3,:]
#         points_reversed = tf.stack([p3,p4,p1,p2])

#         return image_tensor_reversed, points_reversed

#     return tf.cond(is_flipped, flip, lambda: (image_tensor, points_tensor))
