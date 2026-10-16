



import cv2
import sys


def main():
    print("================================")
    print("智慧人臉辨識出勤管理系統")
    print("攝影機測試程式")
    print("================================")
    print("正在開啟攝影機，請稍候...")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("錯誤：無法開啟攝影機！")
        print("請檢查攝影機是否已連接，或是否被其他程式使用。")
        sys.exit(1)

    print("攝影機開啟成功！")
    print("按鍵盤 Q 即可結束測試。")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("錯誤：無法取得攝影機畫面。")
                break

            cv2.putText(
                frame,
                "Camera Test",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Smart Attendance - Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("已結束攝影機測試。")
                break

    except Exception as error:
        print(f"執行時發生錯誤：{error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("攝影機資源已釋放。")


if __name__ == "__main__":
    main()
