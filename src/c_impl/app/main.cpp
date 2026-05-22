#include <QApplication>
#include <QDebug>

#include "app/mainwindow.h"

#include <cstdio>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace {

void logToStderr(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    Q_UNUSED(type);
    Q_UNUSED(context);
    const QByteArray local = msg.toLocal8Bit();
    fprintf(stderr, "%s\n", local.constData());
    fflush(stderr);
}

void attachParentConsoleIfPresent()
{
#ifdef _WIN32
    if (GetConsoleWindow() != nullptr)
        return;
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        (void)freopen("CONOUT$", "w", stdout);
        (void)freopen("CONOUT$", "w", stderr);
        (void)freopen("CONIN$", "r", stdin);
    }
#endif
}

} // namespace

int main(int argc, char *argv[]) {
    attachParentConsoleIfPresent();
    qInstallMessageHandler(logToStderr);

    QApplication app(argc, argv);
    qDebug() << "tracker started (qDebug -> stderr / parent console)";

    MainWindow w;
    w.show();

    return app.exec();
}
