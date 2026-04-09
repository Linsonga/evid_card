/**
 * WeChatAPI.js
 */

import axios from 'axios';
import FormData from 'form-data';
import { promises as fs } from 'fs';
import path from 'path';
import logger from '../utils/logger.js';

/**
 * 微信公众号API服务
 * 封装微信公众平台的API调用，包括access_token管理、图片上传、文章发布等
 */
class WeChatAPI {
  constructor(appId, appSecret) {
    this.appId = appId;
    this.appSecret = appSecret;
    this.accessToken = null;
    this.tokenExpireTime = 0;
    
    logger.debug('WeChatAPI initialized', { appId });
  }

  /**
   * 获取访问令牌(Access Token)
   * 自动处理token缓存和刷新
   * @returns {Promise<string>} Access Token
   */
  async getAccessToken() {
    const now = Date.now();
    
    // 如果token还没过期，直接返回缓存的token
    if (this.accessToken && now < this.tokenExpireTime) {
      logger.debug('使用缓存的access_token');
      return this.accessToken;
    }

    try {
      logger.debug('获取新的access_token');
      const response = await axios.get('https://api.weixin.qq.com/cgi-bin/token', {
        params: {
          grant_type: 'client_credential',
          appid: this.appId,
          secret: this.appSecret
        },
        timeout: 10000
      });

      if (response.data.access_token) {
        this.accessToken = response.data.access_token;
        // 提前60秒过期，避免边界情况
        this.tokenExpireTime = now + (response.data.expires_in - 60) * 1000;
        
        logger.info('access_token获取成功', {
          expiresIn: response.data.expires_in 
        });
        
        return this.accessToken;
      } else {
        throw new Error(`获取Access Token失败: ${response.data.errmsg || '未知错误'}`);
      }
    } catch (error) {
      if (error.response) {
        const errorData = error.response.data;
        throw new Error(`Access Token请求失败: ${errorData.errmsg || error.message}`);
      } else {
        throw new Error(`Access Token网络请求失败: ${error.message}`);
      }
    }
  }

  /**
   * 上传封面图片
   * @param {string} imagePath 图片文件路径
   * @returns {Promise<string>} 媒体ID
   */
  async uploadCoverImage(imagePath) {
    const accessToken = await this.getAccessToken();
    
    try {
      // 检查文件是否存在
      const stats = await fs.stat(imagePath);
      if (!stats.isFile()) {
        throw new Error('指定路径不是有效文件');
      }
      
      // 检查文件大小（微信要求缩略图不超过64KB，这里放宽到1MB）
      if (stats.size > 1024 * 1024) {
        throw new Error('图片文件过大，请使用小于1MB的图片');
      }
      
      // 读取图片文件
      const imageBuffer = await fs.readFile(imagePath);
      const formData = new FormData();
      
      // 根据文件扩展名确定Content-Type
      const ext = path.extname(imagePath).toLowerCase();
      let contentType = 'image/jpeg';
      if (ext === '.png') contentType = 'image/png';
      else if (ext === '.gif') contentType = 'image/gif';
      else if (ext === '.webp') contentType = 'image/webp';
      
      formData.append('media', imageBuffer, {
        filename: path.basename(imagePath),
        contentType: contentType
      });

      logger.debug('开始上传封面图', { 
        path: imagePath, 
        size: stats.size, 
        contentType 
      });

      const response = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/material/add_material?access_token=${accessToken}&type=thumb`,
        formData,
        { 
          headers: formData.getHeaders(),
          timeout: 30000
        }
      );

      if (response.data.media_id) {
        logger.info('封面图上传成功', { 
          mediaId: response.data.media_id,
          url: response.data.url
        });
        return response.data.media_id;
      } else {
        throw new Error(`封面图上传失败: ${response.data.errmsg || '未知错误'}`);
      }
    } catch (error) {
      if (error.code === 'ENOENT') {
        throw new Error(`图片文件不存在: ${imagePath}`);
      } else if (error.response) {
        const errorData = error.response.data;
        throw new Error(`封面图上传失败: ${errorData.errmsg || error.message}`);
      } else {
        throw new Error(`封面图上传请求失败: ${error.message}`);
      }
    }
  }

  // 新增：支持多图文发布
  async publishMultiArticles(articles) {
    const accessToken = await this.getAccessToken();
    const draftArticles = articles.map(art => {
      const articleData = {
        title: art.title,
        author: art.author || '',
        digest: this.extractDigest(art.content),
        content: art.content,
        content_source_url: '',
        need_open_comment: 0,
        only_fans_can_comment: 0
      };
      if (art.thumbMediaId) {
        articleData.thumb_media_id = art.thumbMediaId;
      }
      return articleData;
    });

    // 1. 创建草稿
    const draftResponse = await axios.post(
      `https://api.weixin.qq.com/cgi-bin/draft/add?access_token=${accessToken}`,
      { articles: draftArticles },
      { timeout: 30000 }
    );

    if (draftResponse.data.errcode && draftResponse.data.errcode !== 0) {
      throw new Error(`创建草稿失败: ${draftResponse.data.errmsg}`);
    }

    const mediaId = draftResponse.data.media_id;
    logger.info('多图文草稿创建成功', { mediaId, count: articles.length });

    // 2. 发布
    const publishResponse = await axios.post(
      `https://api.weixin.qq.com/cgi-bin/freepublish/submit?access_token=${accessToken}`,
      { media_id: mediaId },
      { timeout: 30000 }
    );

    if (publishResponse.data.errcode && publishResponse.data.errcode !== 0) {
      throw new Error(`发布文章失败: ${publishResponse.data.errmsg}`);
    }

    return {
      success: true,
      publishId: publishResponse.data.publish_id,
      msgId: publishResponse.data.msg_id,
      mediaId
    };
  }

  /**
   * 发布文章（使用草稿+发布流程）
   * @param {Object} options 发布选项
   * @returns {Promise<Object>} 发布结果
   */
  async publishArticle({ title, content, author, thumbMediaId }) {
    // 检查是否为测试环境（通过AppID判断）
    if (this.appId.startsWith('test_')) {
      logger.info('测试模式：模拟文章发布成功');
      const mockMsgId = Date.now().toString();
      const mockPublishId = (Date.now() - 1000).toString();
      const mockUrl = `https://mp.weixin.qq.com/s/example_${mockMsgId}`;
      
      return {
        success: true,
        publishId: mockPublishId,
        msgId: mockMsgId,
        articleUrl: mockUrl,
        mediaId: 'test_media_id'
      };
    }
    
    logger.info('正式发布模式：调用真实微信API', { appId: this.appId });
    const accessToken = await this.getAccessToken();
    logger.info('获取到access_token', { tokenLength: accessToken.length });
    
    logger.info('开始发布文章到微信公众号');
    logger.info('AppID: ' + this.appId);
    logger.info('文章标题: ' + title);
    logger.info('作者: ' + author);
    
    try {
      logger.debug('开始创建草稿');
      
      // 1. 创建草稿
      const articleData = {
        title,
        author: author || '',
        digest: this.extractDigest(content),
        content,
        content_source_url: '',
        need_open_comment: 0,
        only_fans_can_comment: 0
      };
      
      // 只有当thumbMediaId存在且不为null时才添加thumb_media_id字段
      if (thumbMediaId && thumbMediaId !== null && thumbMediaId !== 'null') {
        articleData.thumb_media_id = thumbMediaId;
      }
      
      const draftData = {
        articles: [articleData]
      };
      
      logger.debug('草稿数据: ' + JSON.stringify({
        ...draftData,
        articles: [{
          ...draftData.articles[0],
          content: `${draftData.articles[0].content.substring(0, 100)}...`
        }]
      }, null, 2));
      logger.debug('thumbMediaId: ' + thumbMediaId);

      logger.info('正在创建草稿...');
      const draftResponse = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/draft/add?access_token=${accessToken}`,
        draftData,
        { timeout: 30000 }
      );

      logger.debug('草稿API响应: ' + JSON.stringify(draftResponse.data, null, 2));

      if (draftResponse.data.errcode && draftResponse.data.errcode !== 0) {
        throw new Error(`创建草稿失败: ${draftResponse.data.errmsg}`);
      }

      const mediaId = draftResponse.data.media_id;
      logger.info('草稿创建成功', { mediaId });
      logger.info('草稿创建成功，MediaID: ' + mediaId);

      // 2. 发布草稿
      logger.debug('开始发布草稿');
      logger.info('正在发布草稿到微信公众号...');
      
      const publishResponse = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/freepublish/submit?access_token=${accessToken}`,
        {
        media_id: mediaId,
        },
        { timeout: 30000 }
      );

      logger.debug('发布API响应: ' + JSON.stringify(publishResponse.data, null, 2));

      if (publishResponse.data.errcode && publishResponse.data.errcode !== 0) {
        throw new Error(`发布文章失败: ${publishResponse.data.errmsg}`);
      }

      const publishId = publishResponse.data.publish_id;
      const msgId = publishResponse.data.msg_id;
      logger.info('文章发布提交成功，发布ID: ' + publishId + '，消息ID: ' + msgId);
      
      // 等待一段时间让文章发布完成，然后查询真实的文章URL
      logger.debug('等待文章发布完成...');
      await new Promise(resolve => setTimeout(resolve, 3000)); // 等待3秒
      
      let articleUrl = null;
      try {
        // 查询发布状态获取真实的文章URL
        const statusResult = await this.getPublishStatus(publishId);
        if (statusResult.article_detail && statusResult.article_detail.item && statusResult.article_detail.item.length > 0) {
          articleUrl = statusResult.article_detail.item[0].url;
        }
      } catch (error) {
        logger.warn('获取文章URL失败，使用默认格式', { error: error.message });
        // 如果查询失败，使用备用URL格式
        articleUrl = `https://mp.weixin.qq.com/s/${publishId}`;
      }
      
      // 如果还是没有获取到URL，使用备用格式
      if (!articleUrl) {
        articleUrl = `https://mp.weixin.qq.com/s/${publishId}`;
      }

      logger.info('文章发布成功', { publishId, msgId, articleUrl });

      return {
        success: true,
        publishId,
        msgId,
        articleUrl,
        mediaId
      };

    } catch (error) {
      if (error.response) {
        const errorData = error.response.data;
        throw new Error(`发布文章失败: ${errorData.errmsg || error.message}`);
      } else {
        throw new Error(`发布文章请求失败: ${error.message}`);
      }
    }
  }

  /**
   * 预览文章
   * @param {Object} options 预览选项
   * @returns {Promise<Object>} 预览结果
   */
  async previewArticle({ title, content, author, thumbMediaId, previewOpenId }) {
    try {
      // 检查是否为测试模式（测试OpenID）
      if (previewOpenId === 'test_openid' || previewOpenId.startsWith('test_')) {
        logger.info('测试模式：模拟预览发送成功');
        const mockMsgId = Date.now().toString();
        const mockUrl = `https://mp.weixin.qq.com/s/example_${mockMsgId}`;
        
        return {
          success: true,
          msgId: mockMsgId,
          articleUrl: mockUrl,
          mediaId: 'test_media_id'
        };
      }
      
      // 先创建图文消息素材
      const mediaId = await this.createNewsMedia({ title, content, author, thumbMediaId });
      
      const accessToken = await this.getAccessToken();
      
      const previewData = {
        touser: previewOpenId,
        mpnews: { media_id: mediaId },
        msgtype: 'mpnews'
      };

      logger.debug('发送预览消息', { previewOpenId, mediaId });

      const response = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/message/mass/preview?access_token=${accessToken}`,
        previewData,
        { timeout: 30000 }
      );

      if (response.data.errcode === 0) {
        logger.info('文章预览发送成功', { msgId: response.data.msg_id });
        return {
          success: true,
          msgId: response.data.msg_id,
          mediaId
        };
      } else {
        throw new Error(`文章预览失败: ${response.data.errmsg}`);
      }
    } catch (error) {
      if (error.response) {
        const errorData = error.response.data;
        throw new Error(`文章预览失败: ${errorData.errmsg || error.message}`);
      } else {
        throw new Error(`文章预览请求失败: ${error.message}`);
      }
    }
  }

  /**
   * 创建图文消息素材（用于预览）
   * @param {Object} options 图文消息选项
   * @returns {Promise<string>} 媒体ID
   */
  async createNewsMedia({ title, content, author, thumbMediaId }) {
    const accessToken = await this.getAccessToken();
    
    try {
      const newsData = {
        articles: [{
          title,
          author: author || '',
          digest: this.extractDigest(content),
          content,
          content_source_url: '',
          show_cover_pic: thumbMediaId ? 1 : 0,
          ...(thumbMediaId ? { thumb_media_id: thumbMediaId } : {})
        }]
      };

      const response = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/media/uploadnews?access_token=${accessToken}`,
        newsData,
        { timeout: 30000 }
      );

      if (response.data.media_id) {
        return response.data.media_id;
      } else {
        throw new Error(`创建图文消息失败: ${response.data.errmsg || '未知错误'}`);
      }
    } catch (error) {
      if (error.response) {
        const errorData = error.response.data;
        throw new Error(`创建图文消息失败: ${errorData.errmsg || error.message}`);
      } else {
        throw new Error(`创建图文消息请求失败: ${error.message}`);
      }
    }
  }

  /**
   * 查询发布状态
   * @param {string} msgId 消息ID或发布ID
   * @returns {Promise<Object>} 状态信息
   */
  async getPublishStatus(msgId) {
    logger.info('开始查询发布状态', { msgId, appId: this.appId });
    
    // 检查是否为明确的测试模式（只有以test_开头的msgId才使用模拟数据）
    if (msgId && msgId.toString().startsWith('test_')) {
      logger.info('测试模式：返回模拟状态数据', { msgId });
      return {
        errcode: 0,
        errmsg: 'ok',
        publish_status: 1, // 发布成功
        article_detail: {
          count: 1,
          item: [{
            article_id: msgId,
            title: '测试文章标题',
            author: '测试作者',
            digest: '这是一篇测试文章',
            content: '',
            content_source_url: '',
            url: `https://mp.weixin.qq.com/s/test_${msgId}`,
            publish_time: Math.floor(Date.now() / 1000),
            stat_info: {
              read_num: 0,  // 测试模式不显示虚假阅读量
              like_num: 0,
              comment_num: 0,
              share_num: 0
            }
          }]
        }
      };
    }
    
    // 对于真实的msgId，始终调用真实的微信API
    const accessToken = await this.getAccessToken();
    logger.debug('获取到access_token，准备查询状态', { tokenLength: accessToken.length });
    
    try {
      logger.debug('调用微信API查询发布状态', { 
        msgId, 
        api: 'freepublish/get' 
      });
      
      const response = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/freepublish/get?access_token=${accessToken}`,
        { publish_id: msgId },
        { timeout: 15000 }
      );

      logger.debug('微信API响应', { 
        errcode: response.data.errcode,
        errmsg: response.data.errmsg,
        hasArticleDetail: !!response.data.article_detail
      });

      if (response.data.errcode === 0) {
        logger.info('状态查询成功', { 
          msgId,
          status: response.data.publish_status,
          articleCount: response.data.article_detail?.count || 0
        });
        return response.data;
      } else {
        // 如果是文章不存在或权限问题，返回更友好的错误信息
        if (response.data.errcode === 40007) {
          throw new Error(`文章不存在或已被删除 (错误码: ${response.data.errcode})`);
        } else if (response.data.errcode === 40001) {
          throw new Error(`access_token无效，请检查AppID和AppSecret (错误码: ${response.data.errcode})`);
        } else {
          throw new Error(`查询发布状态失败: ${response.data.errmsg} (错误码: ${response.data.errcode})`);
        }
      }
    } catch (error) {
      logger.error('状态查询失败', { 
        msgId,
        error: error.message,
        isAxiosError: !!error.response
      });
      
      if (error.response) {
        const errorData = error.response.data;
        throw new Error(`微信API调用失败: ${errorData.errmsg || error.message} (HTTP状态: ${error.response.status})`);
      } else if (error.code === 'ECONNABORTED') {
        throw new Error('请求超时，请检查网络连接后重试');
      } else {
        throw new Error(`网络请求失败: ${error.message}`);
      }
    }
  }

  /**
   * 群发图文消息给所有关注者或指定标签用户
   * 流程：draft/add 创建草稿，再调用 message/mass/sendall 群发
   * （与 publishArticle 共用相同草稿创建逻辑，避免 uploadnews 的 media_id 兼容性问题）
   * @param {Object} options 群发选项
   * @param {string} options.title 文章标题
   * @param {string} options.content 文章HTML内容
   * @param {string} [options.author] 作者
   * @param {string} [options.thumbMediaId] 封面图媒体ID
   * @param {boolean} [options.isToAll=true] 是否群发给所有粉丝
   * @param {number} [options.tagId] 标签ID，isToAll=false时必须提供
   * @param {number} [options.sendIgnoreReprint=0] 0:可转载 1:不可转载
   * @returns {Promise<Object>} 群发结果，包含 msgId 和 msgDataId
   */
  async sendAllMessage({ title, content, author, thumbMediaId, isToAll = true, tagId, sendIgnoreReprint = 0 }) {
    if (this.appId.startsWith('test_')) {
      logger.info('测试模式：模拟群发成功');
      return {
        success: true,
        msgId: Date.now().toString(),
        msgDataId: (Date.now() - 1000).toString(),
        mediaId: 'test_media_id'
      };
    }

    logger.info('开始群发文章', { title, isToAll, tagId });
    const accessToken = await this.getAccessToken();

    // 1. 使用与 publishArticle 相同的 draft/add 创建草稿
    const articleData = {
      title,
      author: author || '',
      digest: this.extractDigest(content),
      content,
      content_source_url: '',
      need_open_comment: 0,
      only_fans_can_comment: 0
    };
    if (thumbMediaId) {
      articleData.thumb_media_id = thumbMediaId;
    }

    try {
      const draftResponse = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/draft/add?access_token=${accessToken}`,
        { articles: [articleData] },
        { timeout: 30000 }
      );

      if (draftResponse.data.errcode && draftResponse.data.errcode !== 0) {
        throw new Error(`创建草稿失败: ${draftResponse.data.errmsg} (错误码: ${draftResponse.data.errcode})`);
      }

      const mediaId = draftResponse.data.media_id;
      logger.info('草稿创建成功', { mediaId });

      // 2. 群发草稿
      const filter = isToAll
        ? { is_to_all: true }
        : { is_to_all: false, tag_id: tagId };

      const sendData = {
        filter,
        mpnews: { media_id: mediaId },
        msgtype: 'mpnews',
        send_ignore_reprint: sendIgnoreReprint
      };

      logger.debug('调用群发接口', { filter, mediaId });

      const response = await axios.post(
        `https://api.weixin.qq.com/cgi-bin/message/mass/sendall?access_token=${accessToken}`,
        sendData,
        { timeout: 30000 }
      );

      logger.debug('群发API响应', response.data);

      if (response.data.errcode !== 0) {
        throw new Error(`群发失败: ${response.data.errmsg} (错误码: ${response.data.errcode})`);
      }

      logger.info('群发提交成功', {
        msgId: response.data.msg_id,
        msgDataId: response.data.msg_data_id
      });

      return {
        success: true,
        msgId: String(response.data.msg_id),
        msgDataId: String(response.data.msg_data_id),
        mediaId
      };
    } catch (error) {
      if (error.response) {
        const errorData = error.response.data;
        throw new Error(`群发请求失败: ${errorData.errmsg || error.message}`);
      } else {
        throw new Error(`群发网络请求失败: ${error.message}`);
      }
    }
  }

  /**
   * 从内容中提取摘要
   * @param {string} content 文章内容
   * @returns {string} 摘要
   */
  extractDigest(content) {
    // 移除所有HTML标签、CSS样式和Markdown标记
    let digest = content
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')  // 移除style标签
      .replace(/<[^>]*>/g, '')      // 移除所有HTML标签
      .replace(/[#*`]/g, '')        // 移除Markdown标记
      .replace(/\s+/g, ' ')         // 替换多个空白字符为单个空格
      .trim();
    
    // 微信公众号摘要限制为64个字符以内
    if (digest.length > 60) {
      digest = digest.substring(0, 60) + '...';
    }
    
    logger.debug('生成的摘要:', digest);
    return digest;
  }
}

export default WeChatAPI;