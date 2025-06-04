def percent_inside(inner_box, outer_box):
    x1, y1, x2, y2 = inner_box
    x1g, y1g, x2g, y2g = outer_box

    # Diện tích của inner_box
    inner_area = (x2 - x1) * (y2 - y1)

    # Tọa độ phần giao nhau
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Diện tích phần giao nhau (phần của inner_box nằm trong outer_box)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Tỷ lệ phần trăm diện tích của inner_box nằm trong outer_box
    return inter_area / inner_area if inner_area > 0 else 0

outer_box = (0, 0, 100, 100)
inner_box = (100, 100, 350, 350)
a = percent_inside(inner_box, outer_box)
print(a)