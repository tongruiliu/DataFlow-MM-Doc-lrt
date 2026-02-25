import type { ThemeNote } from 'vuepress-theme-plume'
import { defineNoteConfig } from 'vuepress-theme-plume'

export const MMGuide: ThemeNote = defineNoteConfig({
    dir: 'mm_guide',
    link: '/mm_guide/',
    sidebar: [
        {
            text: 'Basic Info',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'basicinfo',
            items: [
                'intro',
                'framework',
                'install',
            ],
        },
        {
            text: 'Dataflow Image',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'image_understanding',
            items: [
                'install_image_understanding',
                'image_caption_api',
                'image_vqa_api',
                'context_vqa',
                'context_vqa_api',
                'image_gcot',
                'vision_mct_reasoning_pipeline',
                'image_region_caption_pipeline',
                'image_region_caption_pipeline_api',
                'image_scale_caption_pipeline',
                'image_visual_only_mcq_pipeline',
            ],
        },
        {
            text: 'Dataflow Video',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'video_understanding',
            items: [
                'install_video_understanding',
                'video_caption',
                'video_clip_and_filter',
                'video_qa',
                'video_cotqa',
                'video_longvideo_cotqa_api'
            ],
        },
        {
            text: 'Dataflow Audio',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'audio_understanding',
            items: [
                'install_audio_understanding',
                'audio_qa',
                'audio_voice_activity_detection',
                'audio_asr_pipeline'
            ],
        },
        {
            text: 'Dataflow Generation',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'image_video_generation',
            items: [
                'install_image_video_generation',
                {
                    text: 'Image Generation',
                    collapsed: false,
                    icon: 'material-symbols:add-photo-alternate-outline-rounded',
                    items: [
                        'image_generation_api',
                        'image_generation_gpu',
                    ],
                },
                {
                    text: 'Image Editing',
                    collapsed: false,
                    icon: 'material-symbols:palette-outline',
                    items: [
                        'image_editing_api',
                        'image_editing_gpu',
                    ],
                },
            ],
        },
    ]
})
