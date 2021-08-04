from selenium import webdriver
import pytest
import os
import pandas as pd
import tests as tm


@pytest.fixture(scope="session", autouse=True)
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    driver.get("http://localhost:8501")
    yield driver
    driver.close()


def test_upload(driver):
    driver.implicitly_wait(5)
    file_upload = driver.find_element_by_xpath("//input[@type='file']")
    file_upload.send_keys(f"{os.getcwd()}/tests/iris.csv")
    driver.implicitly_wait(5)


def test_num_rows(driver):
    row_count_element = driver.find_element_by_xpath(
        "//*[contains(text(), 'Number of rows')]"
    )
    assert str(tm.iris["num_rows"]) in row_count_element.get_attribute("innerHTML")


def test_num_columns(driver):
    col_count_element = driver.find_element_by_xpath(
        "//*[contains(text(), 'Number of columns')]"
    )
    assert str(tm.iris["num_cols"]) in col_count_element.get_attribute("innerHTML")


def test_dataset_summary(driver):
    table_element = driver.find_element_by_xpath(
        '//*[@id="root"]/div[1]/div/div/div/div/section[2]/div/div[1]/div[8]/div/div'
    )
    html = table_element.get_attribute("innerHTML")
    df = pd.read_html(html, header=0)[0]

    assert driver.find_element_by_xpath("//*[contains(text(), 'Dataset summary')]")
    assert all([col in df.columns for col in tm.iris["col_names"]])
