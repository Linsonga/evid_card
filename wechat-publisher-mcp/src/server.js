#!/usr/bin/env node
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { z } from 'zod';
import WeChatPublisher from './tools/wechat-publisher.js';
import WeChatStatus from './tools/wechat-status.js';
import WeChatAPI from './services/WeChatAPI.js';
import MarkdownConverter from './services/MarkdownConverter.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 简单的日志函数
//const logger = {
//  info: (msg, data) => console.log(`[INFO] ${msg}`, data || ''),
//  error: (msg, error) => console.error(`[ERROR] ${msg}`, error || ''),
//  debug: (msg, data) => process.env.DEBUG && console.log(`[DEBUG] ${msg}`, data || '')
//};
const logger = {
  info: (msg, data) => console.error(`[INFO] ${msg}`, data || ''),
  error: (msg, error) => console.error(`[ERROR] ${msg}`, error || ''),
  debug: (msg, data) => console.error(`[DEBUG] ${msg}`, data || '')
};

// 创建MCP服务器
const server = new McpServer({
  name: "wechat-publisher-mcp",
  version: "1.0.0"
});

// 注册微信发布工具
server.registerTool(
  "wechat_publish_article",
  {
    description: "将文章发布到微信公众号，支持Markdown格式",
    inputSchema: {
      articles: z.array(z.object({
        title: z.string().describe("文章标题"),
        content: z.string().describe("Markdown格式的文章内容"),
        author: z.string().optional().describe("作者名称"),
        coverImagePath: z.string().optional().describe("封面图片本地路径"),
        coverImageUrl: z.string().optional().describe("封面图片URL")
      })).optional(),
//      title: z.string().describe("文章标题"),
//      content: z.string().describe("Markdown格式的文章内容"),
//      author: z.string().describe("作者名称"),
      appId: z.string().describe("微信公众号AppID"),
      appSecret: z.string().describe("微信公众号AppSecret"),
//      coverImagePath: z.string().optional().describe("封面图片路径"),
      previewMode: z.boolean().default(false).describe("是否为预览模式"),
      previewOpenId: z.string().optional().describe("预览用户OpenID")
    }
  },
  async (params) => {
    logger.info(`Publishing article(s)`);

    try {
      const result = await WeChatPublisher.publish(params);
      return result;
    } catch (error) {
      logger.error(`发布失败: ${error.message}`);
      return {
        content: [{
          type: "text",
          text: `❌ 发布失败: ${error.message}`
        }],
        isError: true
      };
    }
  }
);

// 注册保存草稿工具
server.registerTool(
  "wechat_save_draft",
  {
    description: "将文章保存为微信公众号草稿（不发布），用于在后台预览排版效果",
    inputSchema: {
      articles: z.array(z.object({
        title: z.string().describe("文章标题"),
        content: z.string().describe("Markdown格式的文章内容"),
        author: z.string().optional().describe("作者名称"),
        coverImagePath: z.string().optional().describe("封面图片本地路径"),
        coverImageUrl: z.string().optional().describe("封面图片URL")
      })).describe("文章列表（支持多图文，最多8篇）"),
      appId: z.string().describe("微信公众号AppID"),
      appSecret: z.string().describe("微信公众号AppSecret")
    }
  },
  async (params) => {
    logger.info(`Saving ${params.articles?.length || 1} article(s) to draft`);
    try {
      const result = await WeChatPublisher.saveToDraft(params);
      return result;
    } catch (error) {
      logger.error(`保存草稿失败: ${error.message}`);
      return {
        content: [{
          type: "text",
          text: `❌ 保存草稿失败: ${error.message}`
        }],
        isError: true
      };
    }
  }
);

// 注册群发工具
server.registerTool(
  "wechat_mass_send",
  {
    description: "群发图文消息给微信公众号所有关注者或指定标签用户（主动推送到粉丝消息列表，支持多图文）",
    inputSchema: {
      articles: z.array(z.object({
        title: z.string().describe("文章标题"),
        content: z.string().describe("Markdown格式的文章内容"),
        author: z.string().optional().describe("作者名称"),
        coverImagePath: z.string().optional().describe("封面图片本地路径"),
        coverImageUrl: z.string().optional().describe("封面图片URL")
      })).describe("文章列表（支持多图文，最多8篇）"),
      appId: z.string().describe("微信公众号AppID"),
      appSecret: z.string().describe("微信公众号AppSecret"),
      coverImagePath: z.string().optional().describe("全局封面图片路径（兼容单图文）"),
      coverImageUrl: z.string().optional().describe("全局封面图片URL（兼容单图文）"),
      isToAll: z.boolean().default(true).describe("是否群发给所有粉丝，false时需提供tagId"),
      tagId: z.number().optional().describe("标签ID，isToAll=false时必须提供"),
      sendIgnoreReprint: z.number().default(0).describe("0:允许被转载 1:不允许转载")
    }
  },
  async (params) => {
    const { articles, appId, appSecret, coverImagePath, coverImageUrl, isToAll, tagId, sendIgnoreReprint } = params;
    logger.info(`Mass sending ${articles?.length || 0} article(s)`);

    try {
      const wechatAPI = new WeChatAPI(appId, appSecret);
      const processedArticles = [];

      // 遍历处理每一篇文章（Markdown转HTML + 封面图处理）
      for (let i = 0; i < articles.length; i++) {
        const art = articles[i];
        const htmlContent = MarkdownConverter.convertToWeChatHTML(art.content);
        let thumbMediaId = null;

        // 获取封面图（优先使用单篇配置，第1篇兼容全局配置）
        let coverPath = art.coverImagePath || (i === 0 ? coverImagePath : null);
        let coverUrl = art.coverImageUrl || (i === 0 ? coverImageUrl : null);

        // 如果没有本地路径但有URL，尝试下载
        if (!coverPath && coverUrl) {
          logger.info(`第 ${i + 1} 篇：检测到封面图URL，开始下载...`);
          coverPath = await WeChatPublisher.downloadImage(coverUrl);
        }

        // 仍然没有封面图则自动生成（微信接口要求每篇必须有 thumb_media_id）
        if (!coverPath) {
          logger.info(`第 ${i + 1} 篇：未提供封面图，自动生成...`);
          coverPath = await WeChatPublisher.generateCoverImage(art.title, art.content);
        }

        if (coverPath) {
          try {
            thumbMediaId = await wechatAPI.uploadCoverImage(coverPath);
            logger.info(`第 ${i + 1} 篇：封面图上传成功`, { mediaId: thumbMediaId });

            // 清理自动生成的或下载的临时文件
            const isAutoGenOrUrl = !art.coverImagePath && !(i === 0 && coverImagePath);
            if (isAutoGenOrUrl) {
              const fs = await import('fs/promises');
              await fs.default.unlink(coverPath).catch(() => {});
            }
          } catch (e) {
            logger.error(`第 ${i + 1} 篇封面图上传失败: ${e.message}`);
            throw new Error(`第 ${i + 1} 篇群发需要封面图，但上传失败: ${e.message}`);
          }
        } else {
          throw new Error(`第 ${i + 1} 篇文章缺少封面图`);
        }

        processedArticles.push({
          title: art.title,
          content: htmlContent,
          author: art.author || '',
          thumbMediaId
        });
      }

      // 调用底层的发送接口
      const result = await wechatAPI.sendAllMessage({
        articles: processedArticles,
        isToAll: isToAll !== false,
        tagId,
        sendIgnoreReprint: sendIgnoreReprint || 0
      });

      // 拼接成功响应文案
      let text = `✅ 群发成功！共包含 ${articles.length} 篇文章\n\n`;
      articles.forEach((art, idx) => {
        text += `📝 图文 ${idx + 1}: ${art.title}\n`;
      });
      text += `\n📨 消息ID: ${result.msgId}\n`;
      text += `📊 消息数据ID: ${result.msgDataId}\n`;
      text += `🎯 群发范围: ${isToAll !== false ? '所有粉丝' : `标签ID ${tagId} 的粉丝`}\n`;
      text += `\n🎉 文章已推送到粉丝消息列表，粉丝将收到消息通知。`;

      return { content: [{ type: "text", text }] };
    } catch (error) {
      logger.error(`群发失败: ${error.message}`);
      return {
        content: [{ type: "text", text: `❌ 群发失败: ${error.message}` }],
        isError: true
      };
    }
  }
);
//// 注册群发工具
//server.registerTool(
//  "wechat_mass_send",
//  {
//    description: "群发图文消息给微信公众号所有关注者或指定标签用户（主动推送到粉丝消息列表）",
//    inputSchema: {
//      title: z.string().describe("文章标题"),
//      content: z.string().describe("Markdown格式的文章内容"),
//      author: z.string().describe("作者名称"),
//      appId: z.string().describe("微信公众号AppID"),
//      appSecret: z.string().describe("微信公众号AppSecret"),
//      coverImagePath: z.string().optional().describe("封面图片路径"),
//      isToAll: z.boolean().default(true).describe("是否群发给所有粉丝，false时需提供tagId"),
//      tagId: z.number().optional().describe("标签ID，isToAll=false时必须提供"),
//      sendIgnoreReprint: z.number().default(0).describe("0:允许被转载 1:不允许转载")
//    }
//  },
//  async (params) => {
//    const { title, content, author, appId, appSecret, coverImagePath, isToAll, tagId, sendIgnoreReprint } = params;
//    logger.info(`Mass sending article: ${title}`);
//
//    try {
//      const wechatAPI = new WeChatAPI(appId, appSecret);
//      const htmlContent = MarkdownConverter.convertToWeChatHTML(content);
//
//      // 处理封面图：没有提供时自动生成（uploadnews 接口要求必须有 thumb_media_id）
//      let thumbMediaId = null;
//      let coverPath = coverImagePath;
//
//      if (!coverPath) {
//        logger.info('未提供封面图，自动生成...');
//        coverPath = await WeChatPublisher.generateCoverImage(title, content);
//      }
//
//      if (coverPath) {
//        try {
//          thumbMediaId = await wechatAPI.uploadCoverImage(coverPath);
//          logger.info('封面图上传成功', { mediaId: thumbMediaId });
//
//          // 清理自动生成的临时文件
//          if (!coverImagePath) {
//            const fs = await import('fs/promises');
//            await fs.default.unlink(coverPath).catch(() => {});
//          }
//        } catch (e) {
//          logger.error(`封面图上传失败: ${e.message}`);
//          throw new Error(`群发需要封面图，但上传失败: ${e.message}`);
//        }
//      }
//
//      const result = await wechatAPI.sendAllMessage({
//        title,
//        content: htmlContent,
//        author,
//        thumbMediaId,
//        isToAll: isToAll !== false,
//        tagId,
//        sendIgnoreReprint: sendIgnoreReprint || 0
//      });
//
//      let text = `✅ 群发成功！\n\n`;
//      text += `📱 标题: ${title}\n`;
//      text += `👤 作者: ${author}\n`;
//      text += `📨 消息ID: ${result.msgId}\n`;
//      text += `📊 消息数据ID: ${result.msgDataId}\n`;
//      text += `🎯 群发范围: ${isToAll !== false ? '所有粉丝' : `标签ID ${tagId} 的粉丝`}\n`;
//      text += `\n🎉 文章已推送到粉丝消息列表，粉丝将收到消息通知。`;
//
//      return { content: [{ type: "text", text }] };
//    } catch (error) {
//      logger.error(`群发失败: ${error.message}`);
//      return {
//        content: [{ type: "text", text: `❌ 群发失败: ${error.message}` }],
//        isError: true
//      };
//    }
//  }
//);

// 注册状态查询工具
server.registerTool(
  "wechat_query_status",
  {
    description: "查询文章发布状态和统计数据",
    inputSchema: {
      msgId: z.string().describe("消息ID"),
      appId: z.string().describe("微信公众号AppID"),
      appSecret: z.string().describe("微信公众号AppSecret")
    }
  },
  async (params) => {
    const { msgId, appId, appSecret } = params;
    logger.info(`Querying status for message: ${msgId}`);
    
    try {
      // 调用实际的查询逻辑
      const result = await WeChatStatus.query({
        msgId,
        appId,
        appSecret
      });
      
      return result;
    } catch (error) {
      logger.error(`查询失败: ${error.message}`);
      return {
        content: [{
          type: "text",
          text: `❌ 查询失败: ${error.message}`
        }],
        isError: true
      };
    }
  }
);



logger.info('WeChat Publisher MCP Server initialized');

// 启动服务器函数
async function startServer() {
  try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    logger.info('WeChat Publisher MCP Server connected via stdio');
    return server;
  } catch (error) {
    logger.error('Failed to start server', error);
    throw error;
  }
}

// Start server if running directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  startServer().catch(error => {
    logger.error('Failed to start server', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Received SIGINT, shutting down...');
    process.exit(0);
  });
}

// 包装类
class WeChatMCPServer {
  constructor() {
    this.server = server;
  }
  
  async start() {
    try {
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      logger.info('WeChat Publisher MCP Server connected via stdio');
      return this.server;
    } catch (error) {
      logger.error('Failed to start server', error);
      throw error;
    }
  }
}

export default WeChatMCPServer;