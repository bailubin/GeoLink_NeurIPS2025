import fiona
from shapely.geometry import shape, Point, Polygon, LineString
from shapely import box,relate
from shapely.strtree import STRtree
import numpy as np
from shapely import box
from collections import defaultdict
import re
import shapely
import random
import geopandas as gpd

def get_rtree(shp_path):
    c = fiona.open(shp_path)

    feature_list = []
    geom_list = []
    coord_dict={}
    duplicate_num=0

    for feature in c:
        geometry = feature['geometry']
        geom = shape(geometry)
        coord_key=str(geometry['coordinates'])

        if coord_key in coord_dict:
            duplicate_num+=1
        else:
            feature_list.append(feature)
            geom_list.append(geom)
            coord_dict[coord_key]=feature

    c.close()

    rtree = STRtree(geom_list)

    return rtree, feature_list

def coord2pixel(x_geo, y_geo, ds_geo):
    """
    :param x_geo: 输入x地理坐标
    :param y_geo: 输入y地理坐标
    :param ds_geo: 输入仿射地理变换参数
    :return: 返回x,y像素坐标
    """
    y = ((y_geo - ds_geo[3] - ds_geo[4] / ds_geo[1] * x_geo + ds_geo[4] / ds_geo[1] * ds_geo[
        0]) / (ds_geo[5] - ds_geo[4] / ds_geo[1] * ds_geo[2]))
    x = ((x_geo - ds_geo[0] - y * ds_geo[2]) / ds_geo[1])
    return int(x), int(y)

def pixel2coord(x, y, ds_geo):
    x_geo = ds_geo[0]+ds_geo[1]*x + y*ds_geo[2]
    y_geo = ds_geo[3]+ds_geo[4]*x + y*ds_geo[5]
    return x_geo, y_geo

def get_current_shp(current_region, all_shp_rtree, all_shp_list):

    current_shp_indexes = all_shp_rtree.query(current_region).tolist()

    current_geometry, current_feature = [], []
    for current_shp_index in current_shp_indexes:
        shp = all_shp_rtree.geometries.take(current_shp_index)
        if shp.intersects(current_region):
            c_g=shp.intersection(current_region)
            if c_g.geom_type == 'MultiPolygon' or c_g.geom_type == 'MultiLineString':
                single_gs = list(c_g.geoms)
                for single_g in single_gs:
                    current_geometry.append(single_g)
                    current_feature.append(all_shp_list[current_shp_index])
            else:
                current_geometry.append(c_g)
                current_feature.append(all_shp_list[current_shp_index])

    return current_geometry, current_feature

def get_current_shp_unique(current_region, all_shp_rtree, all_shp_list):

    current_shp_indexes = all_shp_rtree.query(current_region).tolist()

    current_geometry, current_feature = [], []
    for current_shp_index in current_shp_indexes:
        shp = all_shp_rtree.geometries.take(current_shp_index)
        if shp.intersects(current_region):
            c_g=shp.intersection(current_region)
            if c_g.geom_type == 'MultiPolygon' or c_g.geom_type == 'MultiLineString':
                single_gs = list(c_g.geoms)
                for single_g in single_gs:
                    current_geometry.append(single_g)
                    current_feature.append(all_shp_list[current_shp_index])
            else:
                current_geometry.append(c_g)
                current_feature.append(all_shp_list[current_shp_index])

    all_tag = [x['properties']['fclass1'] for x in current_feature]
    res_df = gpd.GeoDataFrame(geometry=current_geometry, crs="EPSG:4326")
    res_df['label'] = all_tag

    return res_df

def get_current_shp_batch(current_region, all_shp_rtree, all_shp_list):
    region_area = current_region.area
    region_length = current_region.length
    current_shp_indexes = all_shp_rtree.query(current_region).tolist()

    current_geometry, current_feature = [], []
    for current_shp_index in current_shp_indexes:
        shp = all_shp_rtree.geometries.take(current_shp_index)
        if shp.intersects(current_region):
            c_g=shp.intersection(current_region)
            if c_g.geom_type == 'Polygon' and c_g.area / region_area < 0.005:
                continue
            if c_g.geom_type == 'LineString' and c_g.length/region_length < 0.01:
                continue
            if c_g.geom_type == 'MultiPolygon' or c_g.geom_type == 'MultiLineString':
                continue
            else:
                current_geometry.append(c_g)
                current_feature.append(all_shp_list[current_shp_index])

    return current_geometry, current_feature

def get_current_shp_sent(current_region, all_shp_rtree, all_shp_list):
    region_area = current_region.area

    current_shp_indexes = all_shp_rtree.query(current_region).tolist()
    current_geometry, current_feature = [], []
    for current_shp_index in current_shp_indexes:
        shp = all_shp_rtree.geometries.take(current_shp_index)
        if shp.intersects(current_region):
            c_g=shp.intersection(current_region)
            if c_g.area / region_area < 0.01:
                continue
            if c_g.geom_type == 'MultiPolygon' or c_g.geom_type == 'MultiLineString':
                single_gs = list(c_g.geoms)
                for single_g in single_gs:
                    current_geometry.append(single_g)
                    current_feature.append(all_shp_list[current_shp_index])
            else:
                current_geometry.append(c_g)
                current_feature.append(all_shp_list[current_shp_index])

    return current_geometry, current_feature

def get_spatial_relation(all_geo, all_feat):

    geo_num=all_geo.shape[0]
    r_m=np.zeros((geo_num,geo_num)).astype(int)
    for i in range(geo_num):
        p_i=all_geo[i]
        for j in range(i+1, geo_num):

            p_j=all_geo[j]
            relation=relate(p_j,p_i)
            # 判断空间关系, disjoint 1, toches 2, overlaps 3, contains 4, within 5, equals 0, other 6
            # disjoint
            if re.match("FF.{1,2}FF.{4,8}",relation) is not None:
                r_m[i,j]=1
                r_m[j,i]=1
            #toches
            elif re.match("F[0,1,2,T].{7,14}",relation) is not None \
            or re.match("F.{2,4}[0,1,2,T].{5,10}",relation) is not None \
            or re.match("F.{3,6}[0,1,2,T].{4,8}",relation) is not None:
                r_m[i,j]=2
                r_m[j,i]=2
            # contain
            elif re.match("[0,1,2,T].{5,10}FF.{1,2}", relation) is not None:
                r_m[i, j] = 5
                r_m[j, i] = 4
            # within
            elif re.match("[0,1,2,T].{1,2}F.{2,4}F.{3,6}", relation) is not None:
                r_m[i, j] = 4
                r_m[j, i] = 5
            #overlaps
            elif re.match("[0,1,2,T].{1,2}[0,1,2,T].{6,12}",relation) is not None:
                r_m[i,j]=3
                r_m[j,i]=3
            # unknown or mask
            else:
                r_m[i,j]=6
                r_m[j,i]=6
    return r_m


def get_spatial_relation_samedim(all_geo):

    geo_num=all_geo.shape[0]
    r_m=np.zeros((geo_num,geo_num)).astype(int)
    for i in range(geo_num):
        p_i=all_geo[i]
        for j in range(i+1, geo_num):

            p_j=all_geo[j]
            relation=relate(p_j,p_i) #j的面积比i小
            # 判断空间关系,self 0, disjoint 1, toches 2, overlaps 3, covers 4, coverby 5, equals 6, other 7
            # disjoint
            if re.match("FF.{1,2}FF.{4,8}",relation) is not None:
                r_m[i,j]=1
                r_m[j,i]=1
            #toches
            elif re.match("F[0,1,2,T].{7,14}",relation) is not None \
            or re.match("F.{2,4}[0,1,2,T].{5,10}",relation) is not None \
            or re.match("F.{3,6}[0,1,2,T].{4,8}",relation) is not None:
                r_m[i,j]=2
                r_m[j,i]=2
            # cover
            elif re.match("[0,1,2,T].{5,10}FF.{1,2}", relation) is not None \
            or re.match(".{1,2}[0,1,2,T].{4,8}FF.{1,2}",relation) is not None \
            or re.match(".{3,6}[0,1,2,T].{2,4}FF.{1,2}",relation) is not None \
            or re.match(".{4,8}[0,1,2,T].{1,2}FF.{1,2}",relation) is not None:
                r_m[i, j] = 5
                r_m[j, i] = 4
            # coverby
            elif re.match("[0,1,2,T].{1,2}F.{2,4}F.{3,6}", relation) is not None\
            or re.match(".{1,2}[0,1,2,T]F.{2,4}F.{3,6}",relation) is not None \
            or re.match(".{2,4}F[0,1,2,T].{1,2}F.{3,6}",relation) is not None \
            or re.match(".{2,4}F.{1,2}[0,1,2,T]F.{3,6}",relation) is not None:
                r_m[i, j] = 4
                r_m[j, i] = 5
            #overlaps
            elif re.match("[0,1,2,T].{1,2}[0,1,2,T].{3,6}[0,1,2,T].{2,4}",relation) is not None:
                r_m[i,j]=3
                r_m[j,i]=3
            # equal
            elif re.match("[0,1,2,T].{1,2}F.{2,4}FFF.{1,2}",relation) is not None:
                r_m[i,j]=6
                r_m[j,i]=6
            # other
            else:
                r_m[i,j]=7
                r_m[j,i]=7
    return r_m

def get_spatial_relation_point2other(point_geo, other_geo):
    # only consider disjoint, touch, within
    point_num=point_geo.shape[0]
    other_num=other_geo.shape[0]
    r_m=np.zeros((point_num,other_num)).astype(int)
    for i in range(point_num):
        p_i=point_geo[i]
        for j in range(other_num):

            p_j=other_geo[j]
            relation=relate(p_i,p_j)
            # disjoint 1, toches 2,  within 3, other 0
            # disjoint
            if re.match("FF.{1,2}FF.{4,8}",relation) is not None:
                r_m[i,j]=1
            #toches
            elif re.match("F[0,1,2,T].{7,14}",relation) is not None \
            or re.match("F.{2,4}[0,1,2,T].{5,10}",relation) is not None \
            or re.match("F.{3,6}[0,1,2,T].{4,8}",relation) is not None:
                r_m[i,j]=2
            # within
            elif re.match("[0,1,2,T].{1,2}F.{2,4}F.{3,6}", relation) is not None:
                r_m[i, j] = 3
            # other
            else:
                pass
    return r_m

def get_spatial_relation_line2polygon(line_geo, polygon_geo):
    # only consider disjoint, touch, cover, converby, cross
    line_num=line_geo.shape[0]
    polygon_num=polygon_geo.shape[0]
    r_m=np.zeros((line_num,polygon_num)).astype(int)
    for i in range(line_num):
        p_i=line_geo[i]
        for j in range(polygon_num):

            p_j=polygon_geo[j]
            relation=relate(p_i,p_j)
            # 判断空间关系, disjoint 1, toches 2, cross 3, coverby 4, other 0
            # disjoint
            if re.match("FF.{1,2}FF.{4,8}", relation) is not None:
                r_m[i, j] = 1
            # toches
            elif re.match("F[0,1,2,T].{7,14}", relation) is not None \
                    or re.match("F.{2,4}[0,1,2,T].{5,10}", relation) is not None \
                    or re.match("F.{3,6}[0,1,2,T].{4,8}", relation) is not None:
                r_m[i, j] = 2
            # coverby
            elif re.match("[0,1,2,T].{1,2}F.{2,4}F.{3,6}", relation) is not None \
                    or re.match(".{1,2}[0,1,2,T]F.{2,4}F.{3,6}", relation) is not None \
                    or re.match(".{2,4}F[0,1,2,T].{1,2}F.{3,6}", relation) is not None \
                    or re.match(".{2,4}F.{1,2}[0,1,2,T]F.{3,6}", relation) is not None:
                r_m[i, j] = 4
            # cross
            elif re.match("[0,1,2,T].{1,2}[0,1,2,T].{6,12}", relation) is not None \
            or re.match("[0,1,2,T].{5,10}[0,1,2,T].{2,4}", relation) is not None:
                r_m[i, j] = 3
            # other
            else:
                pass
    return r_m


def generate_points_around_center(polygon, num_points=5):
    center = polygon.centroid
    angle = 360 / num_points

    distances = []
    for x, y in polygon.exterior.coords:
        vertex = Point(x, y)
        # Calculate Euclidean distance from centroid to vertex
        distance = center.distance(vertex)
        distances.append(distance)
    max_radius = max(distances)/2
    min_radius = min(distances)/2

    points = []
    for i in range(num_points):
        theta = np.radians(angle * i)
        radius = random.uniform(min_radius, max_radius)
        x = center.x + radius * np.cos(theta)
        y = center.y + radius * np.sin(theta)
        point = Point(x, y)
        if polygon.contains(point):
            points.append(point)
        else:
            # Adjust point if it is outside the polygon
            flag = 0
            for r in np.linspace(radius, 0, num=10):
                x = center.x + r * np.cos(theta)
                y = center.y + r * np.sin(theta)
                point = Point(x, y)
                if polygon.contains(point):
                    points.append(point)
                    flag = 1
                    break
            if flag == 0:
                x = center.x + (min_radius * np.cos(theta))/100
                y = center.y + (min_radius * np.sin(theta))/100
                point = Point(x, y)
                points.append(point)
    points.append(center)
    return points

def get_absolute_pos(geo_center, region_origin, delta):
    ab_coords=((geo_center-region_origin)/delta).astype(int)
    print(ab_coords)
    if ab_coords[0]==0:
        if ab_coords[1]==0:
            return "bottom left"
        elif ab_coords[1]==1:
            return "left"
        else:
            return "top left"
    elif ab_coords[0]==1:
        if ab_coords[1]==0:
            return "bottom center"
        elif ab_coords[1]==1:
            return "center"
        else:
            return "top center"
    else:
        if ab_coords[1]==0:
            return "bottom right"
        elif ab_coords[1]==1:
            return "right"
        else:
            return "top right"


def get_relation_text(sub_ob_rm, all_feat, sp_code_relation):

    ob_label_dict=defaultdict(list)
    ob_label_dictu=defaultdict(list)
    relations=[]
    labels=[]

    sub_ob_rm_unique=np.unique(sub_ob_rm)
    for sp_code in sub_ob_rm_unique:
        if sp_code!=1 and sp_code!=0 and sp_code!=6:
            sp_relation=sp_code_relation[sp_code]
            pos_i=np.where(sub_ob_rm==sp_code)[0]

            for i in range(pos_i.shape[0]):
                pos=pos_i[i]
                f_tag_ob=all_feat[pos]['properties']['f_tag']
                label_ob=all_feat[pos]['properties'][f_tag_ob]

                if f_tag_ob!='building':
                    ob_label_dict[sp_relation].append(label_ob)
                else:
                    if label_ob!='building':
                        ob_label_dict[sp_relation].append(label_ob+' building')
                    else:
                        ob_label_dict[sp_relation].append('building')
            ob_label_dictu[sp_relation]=list(np.unique(ob_label_dict[sp_relation]))

    # 将地物类别整理成句子
    for key in ob_label_dictu:

        value_list=ob_label_dictu[key]
        ob_label_string=""
        if len(value_list) >1 and 'building' in value_list:
            value_list.remove('building')
            value_list.append('other buildings')

        if len(value_list) ==1:
            ob_label_string=value_list[0]
        else:
            for v_i, value in enumerate(value_list):
                if v_i==len(value_list)-1:
                    ob_label_string=ob_label_string+'and '+value
                else:
                    ob_label_string=ob_label_string+value+', '
        relations.append(key)
        labels.append(ob_label_string)

    return relations, labels

def get_main_class(all_area, all_label):
    all_label_u=list(np.unique(all_label))
    cls_area=[]
    cls_index=[]
    for label in all_label_u:
        cls_area.append(all_area[all_label==label].sum())
        cls_index.append(np.where(all_label==label)[0])

    zip_a_l = zip(cls_area, all_label_u)
    sorted_zip = sorted(zip_a_l, key=lambda x: x[0], reverse=True)
    cls_area, cls_label = zip(*sorted_zip)
    cls_area, cls_label, cls_index = np.array(cls_area), np.array(cls_label), np.array(cls_index)
    return cls_area, cls_label, cls_index

def get_class_text(all_cls):
    cls_text=""
    if len(all_cls)>5:
        all_cls=np.random.choice(all_cls, size=5,replace=False) #选择的元素不能重复
    for i in range(len(all_cls)-1):
        cls_text=cls_text+all_cls[i]+", "
    if len(all_cls)>1:
        cls_text=cls_text+"and "+all_cls[-1]
    else:
        cls_text+=all_cls[-1]
    return cls_text


def get_main_region(all_area, r_m, ab_threshold=0.05, cv_threshold=0.35):
    if all_area.shape[0]==0:
        return []
    if all_area[0]<ab_threshold:
        return []
    main_index=[0]
    for i in range(1, all_area.shape[0]):
        flag=True
        # disjoint(1) or toches(2)
        for ind in main_index:
            # 相交了
            if r_m[i][ind]>2:
                flag=False
                break
        if flag:
            main_index.append(i)
        main_area=all_area[main_index]
        cv=main_area.std()/main_area.mean()
        print(cv)
        if cv>cv_threshold:
            main_index.pop()
            break

    # if all_area[main_index].sum()/all_area.sum()<0.3:
    #     return []

    return main_index

def select_main_region(main_index, main_labels, r_m):
    main_label_u = np.unique(main_labels)
    if main_label_u.shape[0] > 3:
        select_labels = np.random.choice(main_label_u, size=3, replace=False)
    else:
        select_labels = main_label_u

    # 获得各个类别代表区域的索引
    select_index = []
    for label in select_labels:
        # 获得当前类别的所有索引
        cur_index = main_index[np.where(main_labels == label)[0]]
        # 找出当前类别中空间关系最复杂的区域作为代表区域
        represent_index = cur_index[0]
        relation_num = 0
        for ind in cur_index:
            cur_r_m = r_m[ind]
            cur_relation_num = cur_r_m[(cur_r_m != 0) & (cur_r_m != 1) & (cur_r_m != 6)].shape[0]
            if cur_relation_num > relation_num:
                represent_index = ind
        select_index.append(represent_index)
    return select_index, select_labels

def get_info_rich_region(all_area, all_label, r_m, ab_threshold=0.05, cls_threshold=3):
    rich_index=[]
    rich_label=[]
    for i in range(all_area.shape[0]):
        if all_area[i]<ab_threshold:
            break
        else:
            inside_index=np.where(r_m[i]==4)[0]
            inside_label=all_label[inside_index]
            inside_label_u=np.unique(inside_label)
            if inside_label_u.shape[0]>cls_threshold:
                rich_index.append(i)
                rich_label.append(inside_label_u)
    return rich_index, rich_label


def get_main_road(croad_geo, cregion, all_geo, all_label, d_threshold=5e-4, l_threshold=5e-3):
    lng_left, lat_bottom=np.min(cregion.boundary.coords, axis=0)+1e-6
    lng_right, lat_top=np.max(cregion.boundary.coords, axis=0)-1e-6
    cregion = box(lng_left, lat_bottom, lng_right, lat_top, ccw=False)
    cross_inds=[]
    for i in range(len(croad_geo)):
        if croad_geo[i].length>l_threshold and shapely.crosses(croad_geo[i], cregion):
            cross_inds.append(i)

    cross_inds_true=[]
    in_pos=[]
    out_pos=[]
    along_labels=[]
    main_ind=-1
    main_num=-1
    if len(cross_inds)>0:
        for ind in cross_inds:
            road=croad_geo[ind]
            road_in, road_out=road.boundary.geoms[0].coords[0], road.boundary.geoms[-1].coords[0]
            if road_in[0]<lng_left:
                in_p='left'
            elif road_in[0]>lng_right:
                in_p='right'
            elif road_in[1]<lat_bottom:
                in_p='bottom'
            elif road_in[1]>lat_top:
                in_p='top'
            else:
                in_p=''

            if road_out[0]<lng_left:
                out_p='left'
            elif road_out[0]>lng_right:
                out_p='right'
            elif road_out[1]<lat_bottom:
                out_p='bottom'
            elif road_out[1]>lat_top:
                out_p='top'
            else:
                out_p=''

            if in_p!='' and out_p!='' and in_p!=out_p:
                in_pos.append(in_p)
                out_pos.append(out_p)
                cross_inds_true.append(ind)

                distances=[]
                for geo in all_geo:
                    distances.append(shapely.distance(road, geo))
                distances=np.array(distances)
                a_labels=np.unique(all_label[distances<d_threshold])
                along_labels.append(a_labels)
                if a_labels.shape[0]>main_num:
                    main_ind=ind

    return in_pos,out_pos,along_labels,cross_inds_true, main_ind


