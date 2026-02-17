data_chunks = [[10,20,150,30],[40,200,50,60],[70,80,90,1000]]
output_file = 'anomalies_output.txt'
total_anomalies = 0
with open(output_file, 'w') as out_f:
    for chunk in data_chunks:
        for value in chunk:
            if value > 100:
                out_f.write(str(value) + '\\n')
                total_anomalies += 1
        del chunk
result = total_anomalies
print(result)