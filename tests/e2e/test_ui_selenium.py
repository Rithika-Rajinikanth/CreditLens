"""
CreditLens — Selenium Automated End-to-End (E2E) Browser Tests
==============================================================
Launches the CreditLens FastAPI + Gradio server on an ephemeral port,
spawns a headless browser (Chrome or Edge), navigates to the web UI,
and programmatically exercises task selection, episode initialization,
underwriting actions, and reactive DOM updates.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Generator, Optional

import pytest

pytest.importorskip("selenium", reason="selenium is required for E2E browser tests")
pytest.importorskip("gradio", reason="gradio is required for E2E browser tests")

import uvicorn
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from server.app import app


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url() -> Generator[str, None, None]:
    """Spawns the CreditLens server in a background daemon thread."""
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server readiness
    url = f"http://127.0.0.1:{port}"
    ready = False
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                ready = True
                break
        except Exception:
            time.sleep(0.2)

    if not ready:
        pytest.skip(f"Server failed to start on port {port}")

    yield url
    server.should_exit = True


def _create_headless_driver() -> Optional[webdriver.Remote]:
    """Creates a headless Chrome or Edge WebDriver instance."""
    # Try Chrome first
    try:
        chrome_opts = ChromeOptions()
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--window-size=1280,900")
        driver = webdriver.Chrome(options=chrome_opts)
        return driver
    except Exception as e1:
        # Fallback to Edge
        try:
            edge_opts = EdgeOptions()
            edge_opts.add_argument("--headless=new")
            edge_opts.add_argument("--no-sandbox")
            edge_opts.add_argument("--disable-dev-shm-usage")
            edge_opts.add_argument("--disable-gpu")
            driver = webdriver.Edge(options=edge_opts)
            return driver
        except Exception as e2:
            print(f"[WARN] Headless browsers unavailable: Chrome ({e1}), Edge ({e2})")
            return None


class TestSeleniumUI:
    def test_gradio_ui_loads_and_interacts(self, server_url: str):
        driver = _create_headless_driver()
        if driver is None:
            pytest.skip("Neither Chrome nor Edge WebDriver could be initialized in this environment.")

        try:
            driver.get(server_url)
            wait = WebDriverWait(driver, 15)

            # 1. Verify Page Title & Header
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            assert "CreditLens" in driver.title or "CreditLens" in driver.page_source

            # 2. Locate Start Episode Button
            start_btn = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(., 'Start New Episode')]")
                )
            )
            assert start_btn is not None
            start_btn.click()

            # 3. Wait for Applicant Dossier Card to render
            time.sleep(1.5)
            assert "Applicant" in driver.page_source or "FICO" in driver.page_source

            # 4. Trigger AI Auto-Decision Button
            auto_btn = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(., 'AI Decides')]")
                )
            )
            auto_btn.click()
            time.sleep(1.5)

            # 5. Verify Decision History or Portfolio Update
            page_text = driver.page_source
            assert "APPROVE" in page_text or "REJECT" in page_text or "COUNTER" in page_text or "Step" in page_text

        finally:
            driver.quit()
