
def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters,refrence_height_in_pixel):
    return pixel_distance * (refrence_height_in_meters / refrence_height_in_pixel)


def convert_meters_to_pixel_distance(pixel_distance, refrence_height_in_meters,refrence_height_in_pixel):
    return pixel_distance * (refrence_height_in_pixel / refrence_height_in_meters)