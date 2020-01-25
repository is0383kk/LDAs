from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score

"""
DMLDA=ARI:0.649072564656122,ACC0.7457627118644068
vMLDA=ARI:0.5539386265045024,ACC0.6949152542372882
gMLDA=ARI:0.6581790111124785,ACC0.7966101694915254
"""

label = [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,8,8,8,9,9,9]
dmlda = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,6,9,3,3,3,3,3,3,4,4,4,4,4,5,8,5,5,5,3,5,6,6,6,6,6,6,7,7,6,7,7,7,7,7,7,7,7,7,7,9,9,9]
vmlda = [2,7,0,4,0,0,0,3,0,0,0,0,0,8,7,5,6,8,6,9,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,5,7,7,7,7,7,7,7,7,7,9,8,3]
gmlda = [1,1,0,0,0,0,0,0,0,0,0,0,1,1,9,1,7,7,7,7,3,3,3,3,3,4,4,4,4,6,5,5,5,5,5,5,9,6,6,6,6,6,6,7,7,7,2,7,7,7,7,7,7,7,7,7,9,9,9]
vmlda = [8,8,0,0,0,0,0,0,0,0,0,0,4,9,2,6,2,6,4,2,3,3,1,3,3,4,4,4,4,4,5,5,5,5,5,5,5,6,6,0,8,5,6,7,7,7,7,7,7,5,7,7,7,8,5,5,9,9,5]
DARI = adjusted_rand_score(label,dmlda)
DACC = accuracy_score(label,dmlda)
print(f"DMLDA=ARI:{DARI},ACC{DACC}")
vARI = adjusted_rand_score(label,vmlda)
vACC = accuracy_score(label,vmlda)
print(f"vMLDA=ARI:{vARI},ACC{vACC}")
gARI = adjusted_rand_score(label,gmlda)
gACC = accuracy_score(label,gmlda)
print(f"gMLDA=ARI:{gARI},ACC{gACC}")
