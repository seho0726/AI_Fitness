import pymysql

# MySQL 연결
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='ndsw6363!',
    database='health_data',
    charset='utf8mb4',  # 한글 깨짐 방지
)

# 커서 생성
cursor = conn.cursor()

# 쿼리 실행
cursor.execute("SELECT * FROM user_routines")

# 결과 가져오기
rows = cursor.fetchall()
for row in rows:
    print(row)

# 연결 닫기
cursor.close()
conn.close()
