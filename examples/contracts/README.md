# v0.4 公共接口样例

这些 JSON 由 Plugin、Framework、Robot SDK 和 Studio 的接口测试共同读取。
`available: false` 是环境探测结果，不表示 Profile 已经通过运行验收。
Robot 命令只包含轨迹和夹爪目标；stop、hold 使用明确端点。
`scene-evaluation.json` 是 robosuite、LIBERO 等原生 evaluator 的运行证据样例，
不能当作 Workflow、Robot Skill 或 Ability 的业务结果。
