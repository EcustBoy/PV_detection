%%%Step 1 of graduation project
%%%generate label(object-segmentation) based on original aerial image set
%-------------------------------------------------------------------------%
%centroid_pixels:the coordinates of central point
%centroid_pixels=image1_central_point;
%bound_polygon_pixels:the polygon_bounding box of a PV device
%bound_polygon_pixels=image1_vertices_point;
base_path='D:\my_course\graduation_project\AerialImageSolarArray\3385804';% the relative path of original images 
content=dir(strcat(base_path,'\*.tif'));
img_num=length(content);
% id_seq=zeros(img_num,2);
% data_num=length(image_name2);
% flag=0;
% id1=0;
% id2=0;
% for i=1:img_num
%     name_=strsplit(content(i).name,'.');
%     name_=name_{1,1};
%     flag=0;
%     for j=1:data_num
%         if image_name2(j)==string(name_)
%             if flag==0
%                 id1=j;
%                 flag=flag+1;
%             else
%                 id2=j;
%             end
%             id_seq(i,1)=id1;
%             id_seq(i,2)=id2;
%         end
%     end
% end

%1:lan lon
%2:lon lan
%lon<-->x;lan<-->y

%use datasheet1
img_path=strcat(content(1).folder,'\10sfg465970.tif');
img=imread(img_path);
img_label=img;
pixels_num=size(image_central_point1,1);
figure(1);
imshow(img);
hold on;
%先标出GT框中心点
for i=1:pixels_num
    figure(1)
    scatter(round(image_central_point1(i,2)),round(image_central_point1(i,1)),'.b');
%     round(image1_centroid_pixels(i,1))
%     round(image1_centroid_pixels(i,2))
%     disp('----------------')
    hold on;
end
hold on;

%标出GT框各顶点
for k=1:27
    %剔除NAN
    temp=[];
    for p=1:76
        if isnan(image_polygon_vertices1(k,p))
            continue;
        end
        temp(p)=image_polygon_vertices1(k,p);
    end
    num=length(temp);
    
    i1=1;
    for p=1:num/2
        i2=i1+1;
%         disp('------------------------------')
%         disp('k:');
%         k
%         disp('--------')
%         disp('i1,i2:');
%         i1
%         i2
%         disp('-------')
%         round(image_polygon_vertices1(k,i2))
%         round(image_polygon_vertices1(k,i1))
        scatter(round(image_polygon_vertices1(k,i2)),round(image_polygon_vertices1(k,i1)),'.r');
        %scatter(image_polygon_vertices1(k,i2),image_polygon_vertices1(k,i1),'.r');
        img_label(round(image_polygon_vertices1(k,i2)),round(image_polygon_vertices1(k,i1)),:)=[139,0,0];
        hold on;
        if p>1
            x1=image_polygon_vertices1(k,i2);
            y1=image_polygon_vertices1(k,i1);
            x2=image_polygon_vertices1(k,i2-2);
            y2=image_polygon_vertices1(k,i1-2);
%             disp('x1 y1 x2 y2')
%             x1 
%             y1
%             x2 
%             y2
            %k_=(y2-y1)./(x2-x1);%图像坐标系下斜率
%             disp('round(x1),round(x2):')
%             round(x1)
%             round(x2)
%             disp('---------')
            if round(x1)<round(x2)
                k_=(round(y2)-round(y1))./(round(x2)-round(x1));
                for o1=round(x1):round(x2)
                    o2=round(y1)+k_*(o1-round(x1));
                    o2=round(o2);
                    disp('case 1');p
                    disp('o1,o2:');
                    o1
                    o2
                    disp('---------')
                    img_label(o1,o2,:)=[139,0,0];
                end
            elseif round(x1)>round(x2)
                k_=(round(y2)-round(y1))./(round(x2)-round(x1));
                for o1=round(x1):-1:round(x2)
                    o2=round(y1)+k_*(o1-round(x1));
                    o2=round(o2);
                    disp('case 2');p
                    disp('round(x1).round(x2):');
                    [round(x1),round(x2)]
                    disp('o1,o2:');
                    o1
                    o2
                    disp('---------')
                    img_label(o1,o2,:)=[139,0,0];
                end
            else round(x1)==round(x2)
                if round(y1)>round(y2)
                    for o2=round(y1):-1:round(y2)
                        img_label(round(x1),o2,:)=[139,0,0];
                    end
                elseif round(y1)<round(y2)
                    for o2=round(y1):1:round(y2)
                        img_label(round(x1),o2,:)=[139,0,0];
                    end                  
                end
            end
            %由当前已标顶点标出GT框边界线
            L1=line([round(x1),round(x2)],[round(y1),round(y2)]);
            set(L1,'Color','r');
            hold on;
            
            %最后一个顶点
            if p==num/2 
                %给定的单个多边形GT框数据中可能
                if (image_polygon_vertices1(k,2)==x1)&&(image_polygon_vertices1(k,1)==y1)
                    continue;
                end
                
                k_=(round(image_polygon_vertices1(k,1))-y1)./(round(image_polygon_vertices1(k,2))-x1);%图像坐标系下斜率
                for o1=round(x1):round(image_polygon_vertices1(k,2))
                    o2=y1+k_*(o1-x1);
                    o2=round(o2);
                    img_label(o1,o2,:)=[139,0,0];
                end
                
                L2=line([x1,image_polygon_vertices1(k,2)],[y1,image_polygon_vertices1(k,1)]);
                set(L2,'Color','r')
                text(x1,y1,[num2str(k)]);
            end
        end
        hold on;
        i1=i1+2;
       
    end
end
figure(2);
imshow(img_label)
%imwrite(img_label,'D:\my_course\graduation_project\AerialImageSolarArray\3385804\data_preprocess\label_1.jpg')







            




