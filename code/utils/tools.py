def crop(input, target_shape):
    """

    :param input: image in channels first format
    :param target_shape: only using height and width
    :return: cropped input
    """
    input_height, input_width = input.shape[-2:]
    diff_y = (input_height - target_shape[-2]) // 2
    diff_x = (input_width - target_shape[-1]) // 2
    return input[
           ..., diff_y: (diff_y + target_shape[-2]), diff_x: (diff_x + target_shape[-1])
           ]