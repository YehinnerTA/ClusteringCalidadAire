// main.js
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs'); // Necesario para fs.existsSync
const util = require('util'); // Para formatear logs

// Redirigir console.log y console.error a un archivo de log
// Asegúrate de que app esté disponible aquí. Si no, mueve esto después de app.whenReady()
// o usa una ruta fija para el log en desarrollo y userData en producción.
// Para simplicidad, vamos a hacerlo de forma que funcione en ambos contextos
// PERO ten en cuenta que app.getPath('userData') solo funciona después de que el evento 'ready' de app se emite.
// Una mejor forma es inicializar el log DESPUÉS de app.ready o usar un logger más robusto.

let logStream; // Declarar fuera para que sea accesible

function initializeLogging() {
    const logFileDir = app.isPackaged ? path.join(app.getPath('userData'), 'logs') : __dirname;
    if (!fs.existsSync(logFileDir)) {
        fs.mkdirSync(logFileDir, { recursive: true });
    }
    const logFilePath = path.join(logFileDir, 'app-debug.log');
    logStream = fs.createWriteStream(logFilePath, { flags: 'a' }); // 'a' para append

    const originalConsoleLog = console.log;
    const originalConsoleError = console.error;

    console.log = function(...args) {
        const message = util.format(...args) + '\n';
        originalConsoleLog.apply(console, args); // Sigue escribiendo a la consola normal (útil en desarrollo)
        if (logStream) logStream.write(`[LOG] ${new Date().toISOString()}: ${message}`); // Y escribe al archivo de log
    };

    console.error = function(...args) {
        const message = util.format(...args) + '\n';
        originalConsoleError.apply(console, args); // Sigue escribiendo a la consola normal
        if (logStream) logStream.write(`[ERROR] ${new Date().toISOString()}: ${message}`); // Y escribe al archivo de log
    };

    console.log(`--- NUEVA SESIÓN DE LOG ---`);
    console.log(`Ruta del archivo de log: ${logFilePath}`);
    console.log(`app.isPackaged: ${app.isPackaged}`);
    console.log(`app.getPath('userData'): ${app.getPath('userData')}`);
}

// El resto de tus require:
// const { PythonShell } = require('python-shell');
const tcpPortUsed = require('tcp-port-used');
const child_process = require('child_process');

// ... (el resto de tu main.js: mainWindow, pythonProcess, PY_PORT, getPythonPath, createPythonProcess, etc.) ...
// ... hasta el final del archivo ...

// Modifica el inicio de app.whenReady() para inicializar el logging
app.whenReady().then(() => {
    initializeLogging(); // <--- LLAMA A INICIALIZAR LOGS AQUÍ
    console.log('Evento app.whenReady() disparado.');
    console.log('Iniciando servidor Python...');
    createPythonProcess();
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            console.log('Evento app.activate disparado, creando nueva ventana.');
            createWindow();
        }
    });
});