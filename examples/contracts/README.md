# v0.4 public interface samples

[English](README.md) | [简体中文](README.zh-CN.md)

These JSON files are read jointly by the interface tests of the Plugin,
Framework, Robot SDK, and Studio. `available: false` is the result of
environment probing and does not mean the Profile has passed runtime
acceptance. Robot commands contain only trajectory and gripper targets; stop
and hold use explicit endpoints. `scene-evaluation.json` is a sample of
runtime evidence from native evaluators such as robosuite and LIBERO; it must
not be treated as a business result of a Workflow, Robot Skill, or Ability.
