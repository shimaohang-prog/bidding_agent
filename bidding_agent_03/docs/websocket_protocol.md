# WebSocket 协议

地址：`/api/v1/ws/chat`。认证只使用 HttpOnly Cookie；服务端校验 `Origin`，不得把长期 JWT 放进 URL。

客户端事件均含 `type`、`request_id`、`conversation_id`：

- `ask`：另含 `question`、`client_message_id`、可选 `file_ids`。
- `stop`：按 `request_id` 停止，仅任务所有者有权操作。
- `resume`：另含 `last_seq`，重放 Redis Stream 中更大的序号。
- `ping`：应用层心跳。

服务端事件：`ack`、`status`、`token`、`citations`、`done`、`cancelled`、`error`、`pong`。除 `ack`/`pong` 外均写入 Redis Stream 并获得单调递增 `seq`。正常顺序是：

```text
ack
  ↓
status(planning)
  ↓
status(retrieving)
  ↓
status(reranking)
  ↓
status(generating)
  ↓
token ... → citations → done
```

`done.payload` 含 `message_id`、完整 `answer`、`usage`、`latency_ms` 和 `final_seq`。客户端必须按 `seq` 去重。事件已过 Redis TTL 且 MySQL 显示存在更高序号时，服务端返回 `STREAM_EXPIRED`，客户端应读取历史消息。
