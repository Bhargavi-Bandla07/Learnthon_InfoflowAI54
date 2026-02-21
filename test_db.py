import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="32939",
        database="company_db"
    )
    print("Connected ✅")
    conn.close()

except mysql.connector.Error as e:
    print("Error:", e)