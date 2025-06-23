# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8
# export FLASK_APP=serve.py
# pip install -r requirements_apps.txt --user
# python -m flask run --host=0.0.0.0 --port=8888

mkdir ~/.streamlit
echo "[browser]" > ~/.streamlit/config.toml
echo "gatherUsageStats = true" >> ~/.streamlit/config.toml
echo "serverAddress = \"0.0.0.0\"" >> ~/.streamlit/config.toml
echo "serverPort = 8888" >> ~/.streamlit/config.toml
echo "[server]" >> ~/.streamlit/config.toml
echo "port = 8888" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml
echo "maxMessageSize = 250" >> ~/.streamlit/config.toml


streamlit run streamlit_app.py
