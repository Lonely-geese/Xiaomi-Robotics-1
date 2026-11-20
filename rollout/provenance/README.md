# 来源与整理范围

`source_files.json` 记录 2026-11-20 从 32.160 提取的源文件路径、长度和原始 SHA-256。控制入口取自实际运行进程指定的备份目录，控制辅助模块取自当前工作目录。`live-control-README.txt` 和 `live-runtime-dependencies.json` 保存现场说明及原路径依赖信息。

为支持仓库独立检出，整理版本调整了模型包、SDK、复位文件和控制运行时的定位方式。`control/common.py`、`runtime/model_profile.py`、协议适配器及 VR 导入路径使用本目录结构。模型实现、控制调度、夹爪保持、录制及人工接管逻辑来自现场实现。`control/runtime_dependencies.json` 校验整理版本的依赖，原始文件哈希用于追溯。

录制集成测试来自现场 `steam_rollout/tests/test_xr1_async_rollout.py`。所需的 `FakeObservationSource` 从同目录 `test_steam_rollout.py` 提取为 `control/recording_fixture.py`，去除了对另一套采集程序的测试依赖。

已执行的检查包括 24 项 unittest、完整模拟录制写盘、模型身份与代码校验、控制端 dry-run、Python 编译、Shell 语法和 systemd 服务文件校验。检查在 32.160 的独立临时目录完成。真机动作和新目录下的模型服务联合启动需在现场安排允许复位的时段验证。
